"""WAM policy facade: one obs→action entry point over the two executors.

``WAMPolicy`` is the seam between the server (which hands it preprocessed
observations) and the execution mechanism (which schedules engine calls):

- sync mode (default): :class:`SyncInferenceExecutor` — blocking
  buffer-and-replan with a bounded execution horizon.
- async mode: :class:`AsyncInferenceExecutor` — double-buffered background
  inference overlapping generation with execution.

The executor is chosen once at construction from the normalized async
config; per-step dispatch is plain delegation.
"""

from collections import deque

import numpy as np

from openwam.deploy.engine import BaseInferenceEngine
from openwam.deploy.executors import (
    AsyncInferenceExecutor,
    SyncInferenceExecutor,
    normalize_execution_config,
)


class WAMPolicy:
    """Unified policy facade over the sync / async execution mechanisms.

    Args:
        engine: Inference engine that generates action chunks.
        cfg: Root config, retained for policy-level consumers.
        execution_config: ExecutionConfig-like. ``inference_horizon`` applies
            to both modes; ``inference_delay_steps`` applies only to async.
    """

    def __init__(self, engine: BaseInferenceEngine, cfg, execution_config=None):
        self.cfg = cfg
        self.engine = engine
        inf = getattr(cfg, "inference", None)
        self._ikv_rgb_enabled = bool(getattr(inf, "ikv_rgb_enabled", False))
        self._ikv_dino_model_path = getattr(inf, "ikv_dino_model_path", None)
        self._ikv_require_dino = bool(getattr(inf, "ikv_require_dino", False))
        history_frames = int(getattr(inf, "ikv_history_frames", 16))
        if history_frames < 1:
            raise ValueError("ikv_history_frames must be positive")
        self._ikv_history = deque(maxlen=history_frames)
        self._ikv_dino_history = deque(maxlen=history_frames)
        self._ikv_temporal_stride = 4
        if self._ikv_rgb_enabled:
            backbone = getattr(getattr(engine, "architecture", None), "video_backbone", None)
            if backbone is None or not getattr(backbone, "_is_ti2v", False):
                raise ValueError("OpenWAM RGB IKV requires a Wan TI2V checkpoint")
            self._ikv_temporal_stride = int(getattr(backbone, "_temporal_compression", 4))
            if self._ikv_temporal_stride < 1:
                raise ValueError("Wan temporal compression must be positive")

        self._execution_config = normalize_execution_config(execution_config)
        self._async = self._execution_config.enabled
        if self._async:
            self._executor = AsyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
                inference_delay_steps=self._execution_config.inference_delay_steps,
            )
        else:
            self._executor = SyncInferenceExecutor(
                engine=engine,
                inference_horizon=self._execution_config.inference_horizon,
            )

    def predict_action(self, obs: dict) -> np.ndarray:
        """Return the next action for the given (already preprocessed) observation.

        The final legality projection for two-point command dims
        (``architecture.binary_command_dims``, from the CKPT's dataloader.binary_action_dims) runs
        HERE — after all executor arithmetic. The normalizer already emits exact ±1 for those dims,
        and this final boundary also protects engines or checkpoints that emit
        values between the two legal commands. Threshold 0.5 preserves the
        downstream command contract for the WS server and direct consumers.
        """
        action = self._executor.predict_action(self._build_conditions(obs))
        dims = getattr(getattr(self.engine, "architecture", None), "binary_command_dims", ()) or ()
        if dims:
            action = np.array(action)
            for d in dims:
                if d >= action.shape[-1]:
                    raise ValueError(
                        f"binary_command_dims includes {d} but the action is {action.shape[-1]}-D; "
                        "the ckpt config and the served action width disagree."
                    )
                action[..., d] = np.where(action[..., d] > 0.5, 1.0, -1.0)
        return action

    def reset(self):
        """Clear executor state between episodes."""
        self._ikv_history.clear()
        self._ikv_dino_history.clear()
        self._executor.reset()

    def shutdown(self):
        """Release executor resources (background threads in async mode)."""
        self._executor.shutdown()

    def _build_conditions(self, obs: dict) -> dict:
        """Assemble inference conditions from the current observation.

        Populates the engine-facing fields (``first_frame_image``,
        ``prompt``) from the server-preprocessed observation so the
        pipeline receives images without any further client-side work.
        """
        conditions = {
            "observation": obs,
        }
        img = obs.get("image")
        if img is not None:
            # Single first frame — pipeline expects list[PIL.Image]
            conditions["first_frame_image"] = [img]
            if self._ikv_rgb_enabled:
                self._ikv_history.append(img)
                observed_history = list(self._ikv_history)
                # The causal WAN VAE emits frame endpoints 0,4,8,... .
                # Left-pad with the oldest real frame so the newest real RGB
                # observation is always the final latent endpoint.
                stride = self._ikv_temporal_stride
                pad = (stride - (len(observed_history) - 1) % stride) % stride
                history = [observed_history[0]] * pad + observed_history
                conditions["first_frame_image"] = history
                conditions["ikv_rgb_images"] = history
                inf = self.cfg.inference
                conditions["ikv_patch_capacity"] = int(getattr(inf, "ikv_patch_capacity", 512))
                conditions["ikv_top_k"] = int(getattr(inf, "ikv_top_k", 128))
                conditions["ikv_motion_threshold"] = float(getattr(inf, "ikv_motion_threshold", 0.04))
                if obs.get("ikv_dino_features") is not None:
                    import torch

                    self._ikv_dino_history.clear()
                    supplied = torch.as_tensor(obs["ikv_dino_features"])
                    if supplied.shape[0] == len(observed_history):
                        supplied = torch.cat((supplied[:1].expand(pad, *supplied.shape[1:]), supplied))
                    conditions["ikv_dino_features"] = supplied
                elif self._ikv_dino_model_path:
                    from openwam.model.ikv_dino import encode_dino_grid
                    import torch

                    # Encode only the newly observed RGB frame. Prior DINO
                    # indexes remain in the episode ring with their frames.
                    if len(self._ikv_dino_history) < len(observed_history) - 1:
                        self._ikv_dino_history.clear()
                        self._ikv_dino_history.extend(encode_dino_grid(
                            observed_history, checkpoint=self._ikv_dino_model_path, grid=(16, 16)
                        ))
                    else:
                        self._ikv_dino_history.append(encode_dino_grid(
                            [img], checkpoint=self._ikv_dino_model_path, grid=(16, 16)
                        )[0])
                    dino_history = list(self._ikv_dino_history)
                    conditions["ikv_dino_features"] = torch.stack(
                        [dino_history[0]] * pad + dino_history
                    )
                elif self._ikv_require_dino:
                    raise ValueError("RGB IKV requires DINO features or a local DINOv2 checkpoint")
        if obs.get("prompt"):
            conditions["prompt"] = obs["prompt"]
        if "state" in obs and obs["state"] is not None:
            conditions["proprio"] = obs["state"]
        return conditions
