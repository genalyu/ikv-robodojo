"""RGB history patch selection shared by the JAX and PyTorch pi0.5 paths.

Memory frames are named ``memory_000_rgb``, ``memory_001_rgb``, ... in
chronological order. The selector only masks existing SigLIP tokens; it adds
no trainable weights and does not mix RGB index values into K/V content.
"""

from __future__ import annotations

import math


def memory_keys(images):
    keys = sorted(key for key in images if key.startswith("memory_") and key.endswith("_rgb"))
    if keys and keys != [f"memory_{i:03d}_rgb" for i in range(len(keys))]:
        raise ValueError("RGB memory keys must be contiguous and zero-based")
    return keys


def _validate(capacity, top_k, patch_count):
    if capacity < patch_count or top_k < 0:
        raise ValueError("RGB IKV capacity must fit one complete latest frame")


def select_torch(images, valid, patch_count, *, capacity=1024, top_k=256, threshold=0.04,
                 dino=None, reference_dino=None, current_rgb=None,
                 query_usage=None, action_repetition=None):
    """Images [B,T,H,W,3] in [-1,1]; return [B,T,P] visibility."""
    import torch
    import torch.nn.functional as F

    _validate(capacity, top_k, patch_count)
    batch, frames, height, width, channels = images.shape
    side = math.isqrt(patch_count)
    if channels != 3 or side * side != patch_count or valid.shape != (batch, frames):
        raise ValueError("RGB memory images/masks do not match a square patch grid")
    rgb = images.float().reshape(batch, frames, height, width, 3)
    motion_images = torch.zeros_like(rgb[..., :1])
    motion_images[:, 1:] = (rgb[:, 1:] - rgb[:, :-1]).abs().mean(-1, keepdim=True) / 2
    motion = F.adaptive_avg_pool2d(
        motion_images.reshape(batch * frames, height, width, 1).permute(0, 3, 1, 2),
        (side, side),
    ).reshape(batch, frames, patch_count)
    if current_rgb is not None:
        current = torch.as_tensor(current_rgb, device=images.device).float()
        if current.shape != (batch, height, width, 3):
            raise ValueError("current RGB must align with history image shape")
        transition = (current - rgb[:, -1]).abs().mean(-1, keepdim=True) / 2
        transition = F.adaptive_avg_pool2d(
            transition.permute(0, 3, 1, 2), (side, side)
        ).flatten(1)
        motion[:, -1] = torch.maximum(motion[:, -1], transition)
    first = valid & (valid.long().cumsum(1) == 1)
    eligible = ((motion > threshold) | first[:, :, None]) & valid[:, :, None]
    age = torch.arange(frames - 1, -1, -1, device=images.device).float()
    importance = motion / motion.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    importance += torch.exp(-(age + (1 if reference_dino is not None else 0)) / 8).view(1, frames, 1)
    if dino is not None:
        features = torch.as_tensor(dino, device=images.device).float()
        if features.shape[:3] != (batch, frames, patch_count) or features.ndim != 4:
            raise ValueError("DINO index must be [batch,frames,patches,features]")
        latest = features[:, -1] if reference_dino is None else torch.as_tensor(
            reference_dino, device=images.device
        ).float()
        if latest.shape != (batch, patch_count, features.shape[-1]):
            raise ValueError("current DINO reference must be [batch,patches,features]")
        left = F.normalize(features, dim=-1)
        right = F.normalize(latest, dim=-1)
        visual = torch.einsum("btpd,bqd->btpq", left, right).amax(-1).clamp(0, 1)
        importance += visual * (features != 0).any(-1)
    for values, sign in ((query_usage, 1), (action_repetition, -1)):
        if values is not None:
            value = torch.as_tensor(values, device=images.device).float().reshape_as(motion)
            importance += sign * value / value.amax(dim=(1, 2), keepdim=True).clamp_min(1e-12)
    flat_score = importance.flatten(1)
    eligible_flat = eligible.flatten(1)
    rank = torch.argsort(torch.argsort(
        flat_score.masked_fill(~eligible_flat, -torch.inf), dim=1,
        descending=True, stable=True), dim=1)
    protected = (rank < top_k) & eligible_flat
    indices = torch.arange(frames * patch_count, device=images.device, dtype=torch.long)
    random = ((indices * 1103515245 + 12345) % 2147483647).float() / 2147483647
    priority = torch.where(protected, 2 + flat_score, random)
    priority = priority.masked_fill(~eligible_flat, -torch.inf)
    final_rank = torch.argsort(torch.argsort(priority, dim=1, descending=True, stable=True), dim=1)
    return ((final_rank < capacity) & eligible_flat).reshape(batch, frames, patch_count)


def select_jax(images, valid, patch_count, *, capacity=1024, top_k=256, threshold=0.04,
               dino=None, reference_dino=None, current_rgb=None,
               query_usage=None, action_repetition=None):
    """JAX equivalent of the PyTorch selector; shapes are static under JIT."""
    import jax
    import jax.numpy as jnp

    _validate(capacity, top_k, patch_count)
    batch, frames, height, width, channels = images.shape
    side = math.isqrt(patch_count)
    if channels != 3 or side * side != patch_count or valid.shape != (batch, frames):
        raise ValueError("RGB memory images/masks do not match a square patch grid")
    diff = jnp.concatenate([
        jnp.zeros((batch, 1, height, width, 1)),
        jnp.mean(jnp.abs(images[:, 1:] - images[:, :-1]), axis=-1, keepdims=True) / 2,
    ], axis=1)
    if height % side == 0 and width % side == 0:
        pooled = diff.reshape(batch, frames, side, height // side, side,
                              width // side, 1).mean(axis=(3, 5))
    else:
        pooled = jax.image.resize(diff.astype(jnp.float32),
                                  (batch, frames, side, side, 1), method="linear")
    motion = pooled.reshape(batch, frames, patch_count)
    if current_rgb is not None:
        if current_rgb.shape != (batch, height, width, 3):
            raise ValueError("current RGB must align with history image shape")
        transition = jnp.mean(jnp.abs(current_rgb - images[:, -1]),
                              axis=-1, keepdims=True) / 2
        if height % side == 0 and width % side == 0:
            transition = transition.reshape(batch, side, height // side,
                                            side, width // side, 1).mean(axis=(2, 4))
        else:
            transition = jax.image.resize(transition,
                                          (batch, side, side, 1), method="linear")
        motion = motion.at[:, -1].set(jnp.maximum(motion[:, -1],
                                                  transition.reshape(batch, patch_count)))
    first = valid & (jnp.cumsum(valid.astype(jnp.int32), axis=1) == 1)
    eligible = ((motion > threshold) | first[:, :, None]) & valid[:, :, None]
    age = jnp.arange(frames - 1, -1, -1)
    recency = jnp.exp(-(age + (1 if reference_dino is not None else 0)) / 8).reshape(1, frames, 1)
    importance = motion / jnp.maximum(jnp.max(motion, axis=(1, 2), keepdims=True), 1e-12) + recency
    if dino is not None:
        if dino.ndim != 4 or dino.shape[:3] != (batch, frames, patch_count):
            raise ValueError("DINO index must be [batch,frames,patches,features]")
        norm = dino / jnp.maximum(jnp.linalg.norm(dino, axis=-1, keepdims=True), 1e-12)
        latest = norm[:, -1] if reference_dino is None else (
            reference_dino / jnp.maximum(jnp.linalg.norm(reference_dino, axis=-1, keepdims=True), 1e-12)
        )
        if latest.shape != (batch, patch_count, dino.shape[-1]):
            raise ValueError("current DINO reference must be [batch,patches,features]")
        visual = jnp.max(jnp.einsum("btpd,bqd->btpq", norm, latest), axis=-1)
        importance += jnp.clip(visual, 0, 1) * jnp.any(dino != 0, axis=-1)
    for values, sign in ((query_usage, 1), (action_repetition, -1)):
        if values is not None:
            value = jnp.asarray(values).reshape(motion.shape)
            importance += sign * value / jnp.maximum(jnp.max(value, axis=(1, 2), keepdims=True), 1e-12)
    motion_rank = jnp.argsort(
        jnp.argsort(-jnp.where(eligible, importance, -jnp.inf).reshape(batch, -1), axis=1, stable=True), axis=1
    ).reshape(batch, frames, patch_count)
    protected = (motion_rank < top_k) & eligible
    indices = jnp.arange(frames * patch_count, dtype=jnp.uint32)
    random = ((indices * jnp.uint32(1103515245) + jnp.uint32(12345)) % jnp.uint32(2147483647)).astype(jnp.float32) / 2147483647
    retention = jnp.where(protected, 2 + importance, random.reshape(1, frames, patch_count))
    retention = jnp.where(eligible, retention, -jnp.inf)
    flat = retention.reshape(batch, -1)
    rank = jnp.argsort(jnp.argsort(-flat, axis=1, stable=True), axis=1)
    keep = (rank < min(capacity, frames * patch_count)).reshape(batch, frames, patch_count)
    return keep & eligible


def compact_torch_prefix_tokens(tokens, mask, memory_start, memory_tokens, capacity):
    """Gather selected RGB patches before the costly PaliGemma prefill."""
    import torch

    if not memory_tokens or memory_tokens <= capacity:
        return tokens, mask
    batch, length, hidden = tokens.shape
    new_length = length - memory_tokens + capacity
    pos = torch.arange(length, device=tokens.device)
    memory = (pos >= memory_start) & (pos < memory_start + memory_tokens)
    priority = torch.where(memory[None], torch.where(mask, 1, 2), 0)
    positions = torch.argsort(priority, dim=1, stable=True)[:, :new_length].sort(dim=1).values
    return (
        tokens.gather(1, positions[:, :, None].expand(batch, new_length, hidden)),
        mask.gather(1, positions),
    )


def compact_jax_prefix_tokens(tokens, mask, memory_start, memory_tokens, capacity):
    """Fixed-shape per-sample RGB patch gather before JAX prefill."""
    import jax.numpy as jnp

    if not memory_tokens or memory_tokens <= capacity:
        return tokens, mask
    batch, length, _ = tokens.shape
    new_length = length - memory_tokens + capacity
    pos = jnp.arange(length)
    memory = (pos >= memory_start) & (pos < memory_start + memory_tokens)
    priority = jnp.where(memory[None], jnp.where(mask, 1, 2), 0)
    positions = jnp.sort(jnp.argsort(priority, axis=1, stable=True)[:, :new_length], axis=1)
    return (
        jnp.take_along_axis(tokens, positions[:, :, None], axis=1),
        jnp.take_along_axis(mask, positions, axis=1),
    )
