"""Observed RGB motion patches and their separate DINOv2 retention index."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def _visual_relevance(features: torch.Tensor) -> torch.Tensor:
    """Max nonnegative cosine similarity to patches at the latest real t0."""
    flat = features.flatten(0, 1).float()
    reference = features[-1].float()
    present = (flat != 0).any(-1)
    reference = reference[(reference != 0).any(-1)]
    result = flat.new_zeros(len(flat))
    if len(reference):
        left = F.normalize(flat, dim=-1)
        right = F.normalize(reference, dim=-1)
        for start in range(0, len(flat), 256):
            result[start:start + 256] = (left[start:start + 256] @ right.T).amax(-1).clamp(0, 1)
    return result.reshape(features.shape[:2]) * present.reshape(features.shape[:2])


@torch.no_grad()
def select_clean_prefix_patches(
    rgb_frames,
    *,
    clean_latent_frames: int,
    grid_height: int,
    grid_width: int,
    capacity: int,
    top_k: int,
    motion_threshold: float,
    device,
    dino_features=None,
    query_usage=None,
    action_repetition=None,
    seed: int = 0,
    temporal_stride: int = 4,
) -> torch.Tensor:
    """Return original clean video K/V positions that remain visible.

    RGB differences are computed between adjacent raw observations before
    reducing their scores to Wan's causal latent endpoints. Only the first
    observed frame and changed later patches may enter the cache. DINO is a
    sidecar index, never patch content.
    """
    if not rgb_frames or clean_latent_frames < 1:
        raise ValueError("RGB IKV requires nonempty chronological history")
    if capacity < grid_height * grid_width or top_k < 0 or not 0 <= motion_threshold <= 1:
        raise ValueError("invalid RGB IKV retention settings")
    tensors = []
    for frame in rgb_frames:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("RGB IKV history frames must be HWC RGB")
        tensors.append(torch.as_tensor(array.copy(), device=device).permute(2, 0, 1).float() / 255)
    frames = torch.stack(tensors)
    if temporal_stride < 1:
        raise ValueError("Wan temporal compression must be positive")
    endpoints = torch.arange(clean_latent_frames, device=device) * temporal_stride
    if len(frames) != 1 + temporal_stride * (clean_latent_frames - 1):
        raise ValueError("Wan RGB history must align raw frames to causal VAE endpoints")
    raw = torch.zeros(len(frames), grid_height, grid_width, device=device)
    raw[1:] = F.adaptive_avg_pool2d(
        (frames[1:] - frames[:-1]).abs().mean(1, keepdim=True),
        (grid_height, grid_width),
    )[:, 0]
    blocks = []
    start = 0
    for endpoint in endpoints.tolist():
        blocks.append(raw[start:endpoint + 1].amax(0))
        start = endpoint + 1
    motion = torch.stack(blocks).flatten(1)
    selected = motion > motion_threshold
    selected[0] = True
    if int(selected.sum()) <= capacity:
        return selected.flatten()

    patches = grid_height * grid_width
    age = torch.arange(clean_latent_frames - 1, -1, -1, device=device).float()
    score = motion / motion.amax().clamp_min(1e-12) + torch.exp(-age / 8)[:, None]
    if dino_features is not None:
        dino = torch.as_tensor(dino_features, device=device).float()
        if dino.shape[0] == len(frames) and dino.shape[0] != clean_latent_frames:
            dino = dino.index_select(0, endpoints)
        if dino.ndim == 4:
            dino = F.adaptive_avg_pool2d(dino.permute(0, 3, 1, 2),
                                         (grid_height, grid_width)).flatten(2).transpose(1, 2)
        if dino.ndim != 3 or dino.shape[:2] != (clean_latent_frames, patches):
            raise ValueError("DINO index must align with clean Wan patch grid")
        if not torch.isfinite(dino).all():
            raise ValueError("DINO index must be finite")
        score += _visual_relevance(dino)
    for values, sign in ((query_usage, 1), (action_repetition, -1)):
        if values is not None:
            item = torch.as_tensor(values, device=device).float().reshape_as(score)
            if not torch.isfinite(item).all() or (item < 0).any():
                raise ValueError("retention statistics must be finite and nonnegative")
            score += sign * item / item.amax().clamp_min(1e-12)
    candidates = selected.flatten().nonzero().flatten()
    order = torch.argsort(score.flatten()[candidates], descending=True, stable=True)
    protected = candidates[order[:min(top_k, capacity)]]
    keep = torch.zeros_like(selected.flatten())
    keep[protected] = True
    remaining = candidates[order[min(top_k, capacity):]]
    generator = torch.Generator(device="cpu").manual_seed(seed)
    draw = torch.randperm(len(remaining), generator=generator)[:capacity - len(protected)].to(device)
    keep[remaining[draw]] = True
    return keep
