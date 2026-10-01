"""RGB motion index for the DM05-MEM history prefix.

The index decides which existing visual tokens remain visible. It never changes
the image embeddings or projects RGB statistics into attention keys/values.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from opendm.constants.robot import HISTORY_POOL_SIZE, HISTORY_TOKENS_PER_IMAGE
from opendm.model.dm05.dm05_utils import HISTORY_PAD_TOKEN_ID


def _rgb_unit(images: torch.Tensor) -> torch.Tensor:
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("history RGB must have shape [frames,3,height,width]")
    if not torch.isfinite(images).all():
        raise ValueError("history RGB contains non-finite values")
    source_dtype = images.dtype
    images = images.float()
    if source_dtype == torch.uint8:
        images = images / 255
    elif source_dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ValueError("RGB must be uint8 or floating point in [0,1]")
    if images.numel() and (images.amin() < 0 or images.amax() > 1):
        raise ValueError("RGB values must lie in [0,1]; pass original RGB, not vision pixels")
    return images


@torch.no_grad()
def history_motion_scores(images: torch.Tensor, current_rgb: torch.Tensor | None = None) -> torch.Tensor:
    """Return per-frame, per-patch RGB motion at DM05's 4x4 history grid."""
    rgb = _rgb_unit(images)
    if not len(rgb):
        return rgb.new_zeros((0, HISTORY_TOKENS_PER_IMAGE))
    diff = torch.zeros(len(rgb), 1, *rgb.shape[-2:], device=rgb.device)
    diff[1:, 0] = (rgb[1:] - rgb[:-1]).abs().mean(1)
    scores = F.adaptive_avg_pool2d(diff, (HISTORY_POOL_SIZE, HISTORY_POOL_SIZE))[:, 0].flatten(1)
    if current_rgb is not None:
        current = _rgb_unit(current_rgb)
        if current.shape != rgb[-1:].shape:
            raise ValueError("current RGB and history RGB must share shape")
        transition = F.adaptive_avg_pool2d(
            (current - rgb[-1:]).abs().mean(1, keepdim=True),
            (HISTORY_POOL_SIZE, HISTORY_POOL_SIZE),
        )[:, 0].flatten(1)
        scores[-1] = torch.maximum(scores[-1], transition[0])
    return scores


@torch.no_grad()
def select_history_patches(
    images: torch.Tensor,
    *,
    capacity: int = 128,
    top_k: int = 64,
    motion_threshold: float = 0.04,
    time_scale: float = 8.0,
    seed: int = 0,
    dino_features: torch.Tensor | None = None,
    reference_dino: torch.Tensor | None = None,
    current_rgb: torch.Tensor | None = None,
    query_usage: torch.Tensor | None = None,
    action_repetition: torch.Tensor | None = None,
) -> torch.Tensor:
    """Pick a bounded RGB history index, retaining motion and MEM anchors.

    The first and latest frames are anchor candidates. The top-k remaining
    candidates are protected by motion plus recency; other slots are sampled
    reproducibly, matching N0-TWAM's global top-k/random-remainder policy.
    """
    if capacity < 1 or top_k < 0 or time_scale <= 0 or motion_threshold < 0:
        raise ValueError("invalid RGB IKV retention settings")
    scores = history_motion_scores(images, current_rgb=current_rgb)
    frames, patches = scores.shape
    selected = torch.zeros_like(scores, dtype=torch.bool)
    if not frames:
        return selected
    total = frames * patches
    eligible = scores > motion_threshold
    eligible[0] = True
    if int(eligible.sum()) <= capacity:
        return eligible
    if capacity >= total:
        return eligible
    budget = capacity
    if budget <= 0:
        return selected
    age = torch.arange(frames - 1, -1, -1, device=scores.device).float()
    recency = torch.exp(-(age + (1 if reference_dino is not None else 0)) / time_scale).unsqueeze(1)
    importance = scores / scores.amax().clamp_min(1e-12) + recency
    if dino_features is not None:
        dino = torch.as_tensor(dino_features, device=scores.device).float()
        if dino.ndim != 3 or dino.shape[:2] != (frames, patches):
            raise ValueError("DINO index must be [history_frames,16,feature_dim]")
        if not torch.isfinite(dino).all():
            raise ValueError("DINO index must be finite")
        present = (dino != 0).any(-1)
        reference = dino[-1] if reference_dino is None else torch.as_tensor(
            reference_dino, device=scores.device
        ).float().reshape(-1, dino.shape[-1])
        if not torch.isfinite(reference).all():
            raise ValueError("current DINO reference must be finite")
        latest = F.normalize(reference, dim=-1)
        latest = latest[(reference != 0).any(-1)]
        if len(latest):
            visual = (F.normalize(dino.flatten(0, 1), dim=-1) @ latest.T).amax(-1)
            importance += visual.clamp(0, 1).reshape_as(scores) * present
    for values, sign in ((query_usage, 1), (action_repetition, -1)):
        if values is not None:
            value = torch.as_tensor(values, device=scores.device).float().reshape_as(scores)
            if not torch.isfinite(value).all() or (value < 0).any():
                raise ValueError("retention statistics must be finite and nonnegative")
            importance += sign * value / value.amax().clamp_min(1e-12)
    candidates = eligible
    candidate_flat = candidates.flatten().nonzero().flatten()
    if len(candidate_flat):
        order = torch.argsort(importance.flatten()[candidate_flat], descending=True, stable=True)
        protected = candidate_flat[order[:min(top_k, budget)]]
        selected.flatten()[protected] = True
    budget = capacity - int(selected.sum())
    if budget:
        remaining = (eligible & ~selected).flatten().nonzero().flatten()
        generator = torch.Generator(device="cpu").manual_seed(seed)
        draw = torch.randperm(len(remaining), generator=generator)[:budget].to(remaining.device)
        selected.flatten()[remaining[draw]] = True
    return selected


@torch.no_grad()
def mask_history_prefix(
    input_ids: torch.Tensor,
    history_mask: torch.Tensor | None,
    history_pixels: torch.Tensor | None,
    history_rgb_values: torch.Tensor | None = None,
    current_rgb_values: torch.Tensor | None = None,
    dino_features: torch.Tensor | None = None,
    reference_dino: torch.Tensor | None = None,
    dino_model_path: str | None = None,
    **retention,
) -> torch.Tensor:
    """Mark discarded history placeholders as invisible without moving tokens.

    The original history_mask remains intact so feature scatter still consumes
    exactly the image features produced by the vision tower.
    """
    if history_mask is None or history_pixels is None or not history_mask.any():
        return input_ids
    if history_rgb_values is None:
        raise ValueError("RGB IKV needs original history_rgb_values, not normalized vision pixels")
    if input_ids.shape != history_mask.shape or history_mask.dtype != torch.bool:
        raise ValueError("history_mask must be bool and match input_ids")
    if (dino_features is None or reference_dino is None) and dino_model_path:
        from opendm.model.dm05.ikv_dino import encode_dino_grid

        if current_rgb_values is None:
            raise ValueError("current raw RGB is required for the latest real DINO reference")
        encoded = encode_dino_grid(
            torch.cat((history_rgb_values, current_rgb_values), dim=0), checkpoint=dino_model_path,
            grid=(HISTORY_POOL_SIZE, HISTORY_POOL_SIZE), device=history_rgb_values.device,
        ).flatten(1, 2)
        if dino_features is None:
            dino_features = encoded[:len(history_rgb_values)]
        if reference_dino is None:
            reference_dino = encoded[len(history_rgb_values):]
    result = input_ids.clone()
    offset = 0
    for batch in range(input_ids.shape[0]):
        positions = history_mask[batch].nonzero().flatten()
        if len(positions) % HISTORY_TOKENS_PER_IMAGE:
            raise ValueError("history placeholders are not complete 4x4 image grids")
        frames = len(positions) // HISTORY_TOKENS_PER_IMAGE
        features = None if dino_features is None else dino_features[offset:offset + frames]
        reference = None if reference_dino is None else reference_dino[batch]
        keep = select_history_patches(
            history_rgb_values[offset:offset + frames], dino_features=features,
            reference_dino=reference,
            current_rgb=None if current_rgb_values is None else current_rgb_values[batch:batch + 1],
            **retention
        ).flatten()
        result[batch, positions[~keep]] = HISTORY_PAD_TOKEN_ID
        offset += frames
    if offset != len(history_pixels):
        raise ValueError("history image count does not match history placeholders")
    return result
