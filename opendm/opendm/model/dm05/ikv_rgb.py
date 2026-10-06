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
    images: torch.Tensor, *, capacity: int = 320, top_k: int = 320,
    motion_threshold: float = 0.04, motion_only: bool = False,
    time_scale: float = 8.0, contact_scale: float = 8.0,
    class_recency_scale: float = 8.0, class_threshold: float = 0.9,
    seed: int = 0, dino_features: torch.Tensor | None = None,
    reference_dino: torch.Tensor | None = None,
    current_rgb: torch.Tensor | None = None,
    contact_duration: torch.Tensor | None = None,
) -> torch.Tensor:
    """Select history K/V by T + C + DINO class recency; motion is optional."""
    if (capacity < 1 or top_k < 0 or time_scale <= 0 or contact_scale <= 0
            or class_recency_scale <= 0 or not 0 <= class_threshold <= 1
            or motion_threshold < 0):
        raise ValueError("invalid IKV retention settings")
    motion = history_motion_scores(images, current_rgb=current_rgb)
    frames, patches = motion.shape
    keep = torch.zeros_like(motion, dtype=torch.bool)
    if not frames:
        return keep
    eligible = torch.ones_like(keep)
    if motion_only:
        eligible = motion > motion_threshold
        eligible[0] = True
    age = torch.arange(frames - 1, -1, -1, device=motion.device).float()
    importance = torch.exp(-(age[:, None] + (1 if reference_dino is not None else 0)) / time_scale).expand_as(motion).clone()
    if contact_duration is not None:
        duration = torch.as_tensor(contact_duration, device=motion.device).float().reshape_as(motion)
        if not torch.isfinite(duration).all() or (duration < 0).any():
            raise ValueError("contact duration must be finite and nonnegative")
        importance += 1 - torch.exp(-duration / contact_scale)
    if dino_features is not None:
        dino = torch.as_tensor(dino_features, device=motion.device).float()
        if dino.ndim != 3 or dino.shape[:2] != (frames, patches):
            raise ValueError("DINO index must be [history_frames,16,feature_dim]")
        if not torch.isfinite(dino).all():
            raise ValueError("DINO index must be finite")
        flat = F.normalize(dino.flatten(0, 1), dim=-1)
        times = torch.arange(frames, device=motion.device).repeat_interleave(patches)
        present = dino.flatten(0, 1).ne(0).any(-1)
        if reference_dino is not None:
            reference = torch.as_tensor(reference_dino, device=motion.device).float().reshape(-1, dino.shape[-1])
            if not torch.isfinite(reference).all():
                raise ValueError("current DINO reference must be finite")
            flat = torch.cat((flat, F.normalize(reference, dim=-1)))
            times = torch.cat((times, times.new_full((len(reference),), frames)))
            present = torch.cat((present, reference.ne(0).any(-1)))
        count = frames * patches
        latest = times[:count].clone()
        for start in range(0, count, 128):
            stop = min(start + 128, count)
            match = flat[start:stop] @ flat.T >= class_threshold
            match &= present[start:stop, None] & present[None, :]
            latest[start:stop] = torch.where(match, times[None, :], -1).amax(1).clamp_min(times[start:stop])
        recency = torch.exp(-(latest - times[:count]).float() / class_recency_scale)
        importance += (recency * present[:count]).reshape_as(motion)
    candidates = eligible.flatten().nonzero().flatten()
    if len(candidates) <= capacity:
        keep.flatten()[candidates] = True
        return keep
    order = torch.argsort(importance.flatten()[candidates], descending=True, stable=True)
    protected = candidates[order[:min(top_k, capacity)]]
    keep.flatten()[protected] = True
    remaining = candidates[order[min(top_k, capacity):]]
    budget = capacity - len(protected)
    if budget:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        draw = torch.randperm(len(remaining), generator=generator)[:budget].to(remaining.device)
        keep.flatten()[remaining[draw]] = True
    return keep


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
    if input_ids.shape != history_mask.shape or history_mask.dtype != torch.bool:
        raise ValueError("history_mask must be bool and match input_ids")
    # Every valid patch fits: skip the RGB/DINO index and preserve the full prefix.
    capacity = retention.get("capacity", 320)
    if not retention.get("motion_only", False) and bool((history_mask.sum(1) <= capacity).all()):
        return input_ids
    if history_rgb_values is None:
        raise ValueError("RGB IKV needs original history_rgb_values, not normalized vision pixels")
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
