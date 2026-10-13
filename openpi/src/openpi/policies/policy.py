from collections.abc import Sequence
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        import os
        self._sys2_endpoint=os.environ.get("ROBODOJO_IKV_SYS2_ENDPOINT")
        self._sys2_audit=os.environ.get("ROBODOJO_IKV_AUDIT")
        self._qwen_bank=None
        self._qwen_prefill=nnx_utils.module_jit(model.qwen_prefill) if self._sys2_endpoint else None
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._single_frame_ikv = bool(getattr(model, "ikv_single_frame", False))
            self._sample_actions_single_frame_ikv = (nnx_utils.module_jit(model.sample_actions_single_frame_ikv)
                                                     if self._single_frame_ikv else None)
            self._ikv_state = None
            if self._single_frame_ikv:
                logging.info("PI05 single-frame persistent IKV enabled: visual KV capacity=%d",
                             int(model.ikv_history_capacity))
            self._rng = rng or jax.random.key(0)

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        if self._sys2_endpoint:
            from openpi.models.qwen_archive import QwenArchiveBank
            from openpi.models.sys2 import Sys2Client
            task=str(obs.get("prompt") or "Perform the instructed manipulation task.")
            n=int(self._model.ikv_history_capacity)
            camera_images={name:observation.images[name] for name in _model.IMAGE_KEYS}
            count=len(camera_images)
            side=int((n//count)**0.5)
            if side*side*count!=n:raise ValueError("Qwen visual budget does not match camera grids")
            client=Sys2Client(self._sys2_endpoint,task,timeout=600)
            if self._qwen_bank is None:
                self._qwen_bank=QwenArchiveBank(n,client,task,grid=(side,side),protect_current=False,audit=self._sys2_audit)
            bank=self._qwen_bank
            cache,mask=self._qwen_prefill(observation)
            refs=[]
            for image in camera_images.values():
                image=np.asarray(image)[0]
                rgb=torch.from_numpy(np.array(image)).permute(2,0,1).float()
                rgb=rgb/255 if np.issubdtype(image.dtype,np.integer) else (rgb+1)/2
                refs.extend([client.encode(rgb, grid=(side,side))]*(side*side))
            ids=bank.generation*n+torch.arange(n)
            plan=bank.plan(ids,torch.tensor(refs)[:,None],torch.full((n,),float(bank.generation)))
            ck,cv=cache
            k=torch.from_numpy(np.array(ck[:,:,:n],dtype=np.float32))
            v=torch.from_numpy(np.array(cv[:,:,:n],dtype=np.float32))
            if ck.dtype==jnp.bfloat16:k=k.to(torch.bfloat16)
            elif ck.dtype==jnp.float16:k=k.to(torch.float16)
            if cv.dtype==jnp.bfloat16:v=v.to(torch.bfloat16)
            elif cv.dtype==jnp.float16:v=v.to(torch.float16)
            mk,mv=plan.merge(0,k,v,2)
            selected=(jnp.asarray(mk.float().numpy(),dtype=ck.dtype),jnp.asarray(mv.float().numpy(),dtype=cv.dtype))
            action_cache=(jnp.concatenate((selected[0],ck[:,:,n:]),axis=2),jnp.concatenate((selected[1],cv[:,:,n:]),axis=2))
            actions,_=self._sample_actions_single_frame_ikv(sample_rng_or_pytorch_device,observation,None,external_cache=(action_cache,mask),**sample_kwargs)
            jax.block_until_ready(actions)
            bank.commit(plan,{0:(mk,mv)},1)
        elif not self._is_pytorch_model and self._single_frame_ikv:
            actions, new_state = self._sample_actions_single_frame_ikv(
                sample_rng_or_pytorch_device, observation, self._ikv_state, **sample_kwargs)
            # Publish only after the complete JIT call succeeds.
            self._ikv_state = jax.tree.map(jax.lax.stop_gradient, new_state)
        else:
            actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        outputs = {"state": inputs["state"], "actions": actions}
        model_time = time.monotonic() - start_time
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        outputs = self._output_transform(outputs)
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs

    def reset_ikv(self):
        self._ikv_state = None
        self._qwen_bank = None

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results
