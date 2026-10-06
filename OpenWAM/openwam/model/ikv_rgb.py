"""Observed RGB motion patches and their separate DINOv2 retention index."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


@torch.no_grad()
def select_clean_prefix_patches(
    rgb_frames, *, clean_latent_frames: int, grid_height: int, grid_width: int,
    capacity: int, top_k: int, motion_threshold: float, device,
    dino_features=None, contact_duration=None, motion_only: bool = False,
    time_scale: float = 8.0, contact_scale: float = 8.0,
    class_recency_scale: float = 8.0, class_threshold: float = 0.9,
    seed: int = 0, temporal_stride: int = 4,
) -> torch.Tensor:
    """Choose clean Wan K/V positions with T+C+R; motion gating is optional."""
    if not rgb_frames or clean_latent_frames < 1:
        raise ValueError("RGB IKV requires nonempty chronological history")
    if (capacity < 1 or top_k < 0 or motion_threshold < 0 or temporal_stride < 1
            or min(time_scale, contact_scale, class_recency_scale) <= 0
            or not 0 <= class_threshold <= 1):
        raise ValueError("invalid RGB IKV retention settings")
    tensors = []
    for frame in rgb_frames:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[-1] != 3:
            raise ValueError("RGB IKV history frames must be HWC RGB")
        tensors.append(torch.as_tensor(array.copy(), device=device).permute(2, 0, 1).float() / 255)
    frames = torch.stack(tensors)
    endpoints = torch.arange(clean_latent_frames, device=device) * temporal_stride
    if len(frames) != 1 + temporal_stride * (clean_latent_frames - 1):
        raise ValueError("Wan RGB history must align raw frames to causal VAE endpoints")
    patches = grid_height * grid_width
    eligible = torch.ones((clean_latent_frames, patches), dtype=torch.bool, device=device)
    if motion_only:
        raw = torch.zeros(len(frames), grid_height, grid_width, device=device)
        raw[1:] = F.adaptive_avg_pool2d(
            (frames[1:] - frames[:-1]).abs().mean(1, keepdim=True),
            (grid_height, grid_width))[:, 0]
        blocks = []
        start = 0
        for endpoint in endpoints.tolist():
            blocks.append(raw[start:endpoint + 1].amax(0))
            start = endpoint + 1
        motion = torch.stack(blocks).flatten(1)
        eligible = motion > motion_threshold
        eligible[0] = True
    age = torch.arange(clean_latent_frames - 1, -1, -1, device=device).float()
    score = torch.exp(-age[:, None] / time_scale).expand(clean_latent_frames, patches).clone()
    if contact_duration is not None:
        duration = torch.as_tensor(contact_duration, device=device).float().reshape_as(score)
        if not torch.isfinite(duration).all() or (duration < 0).any():
            raise ValueError("contact duration must be finite and nonnegative")
        score += 1 - torch.exp(-duration / contact_scale)
    if dino_features is not None:
        dino = torch.as_tensor(dino_features, device=device).float()
        if dino.shape[0] == len(frames) and dino.shape[0] != clean_latent_frames:
            dino = dino.index_select(0, endpoints)
        if dino.ndim == 4:
            dino = F.adaptive_avg_pool2d(
                dino.permute(0, 3, 1, 2), (grid_height, grid_width)
            ).flatten(2).transpose(1, 2)
        if dino.ndim != 3 or dino.shape[:2] != (clean_latent_frames, patches):
            raise ValueError("DINO index must align with clean Wan patch grid")
        if not torch.isfinite(dino).all():
            raise ValueError("DINO index must be finite")
        flat = F.normalize(dino.reshape(-1, dino.shape[-1]), dim=-1)
        present = dino.reshape(-1, dino.shape[-1]).ne(0).any(-1)
        times = torch.arange(clean_latent_frames, device=device).repeat_interleave(patches)
        latest = times.clone()
        for start in range(0, len(flat), 128):
            stop = min(start + 128, len(flat))
            match = flat[start:stop] @ flat.T >= class_threshold
            match &= present[start:stop, None] & present[None, :]
            latest[start:stop] = torch.where(match, times[None, :], -1).amax(1).clamp_min(times[start:stop])
        score += (torch.exp(-(latest - times).float() / class_recency_scale) * present).reshape_as(score)
    candidates = eligible.flatten().nonzero().flatten()
    if len(candidates) <= capacity:
        return eligible.flatten()
    order = torch.argsort(score.flatten()[candidates], descending=True, stable=True)
    protected = candidates[order[:min(top_k, capacity)]]
    keep = torch.zeros_like(eligible.flatten())
    keep[protected] = True
    remaining = candidates[order[min(top_k, capacity):]]
    budget = capacity - len(protected)
    if budget:
        generator = torch.Generator(device="cpu").manual_seed(seed)
        draw = torch.randperm(len(remaining), generator=generator)[:budget].to(device)
        keep[remaining[draw]] = True
    return keep
