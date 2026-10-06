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
        self._ikv_kv_state = {}
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
        if self._async and self._ikv_rgb_enabled:
            raise ValueError("persistent KV currently requires synchronous episode execution")
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
        self._ikv_kv_state.clear()
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
                conditions['ikv_kv_state'] = self._ikv_kv_state
                conditions['ikv_current_image'] = img
                conditions['ikv_dino_model_path'] = self._ikv_dino_model_path
                conditions['ikv_dino_features'] = obs.get('ikv_dino_features')
                conditions['ikv_motion_only'] = bool(getattr(self.cfg.inference,'ikv_motion_only',False))
                conditions['ikv_motion_threshold'] = float(getattr(self.cfg.inference,'ikv_motion_threshold',.04))
        if obs.get("prompt"):
            conditions["prompt"] = obs["prompt"]
        if "state" in obs and obs["state"] is not None:
            conditions["proprio"] = obs["state"]
        return conditions
