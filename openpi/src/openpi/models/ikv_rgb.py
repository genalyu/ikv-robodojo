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
    if capacity < 1 or top_k < 0:
        raise ValueError("RGB IKV capacity must be positive and top_k nonnegative")


def select_torch(images, valid, patch_count, *, capacity=1024, top_k=1024,
                 threshold=0.04, motion_only=False, dino=None,
                 reference_dino=None, current_rgb=None, contact_duration=None,
                 class_threshold=0.9, time_scale=8.0, contact_scale=8.0,
                 class_recency_scale=8.0):
    """Select memory K/V positions by T+C+R, optionally reading motion only."""
    import torch
    import torch.nn.functional as F

    _validate(capacity, top_k, patch_count)
    batch, frames, height, width, channels = images.shape
    side = math.isqrt(patch_count)
    if channels != 3 or side * side != patch_count or valid.shape != (batch, frames):
        raise ValueError("RGB memory images/masks do not match a square patch grid")
    if min(time_scale, contact_scale, class_recency_scale) <= 0 or not 0 <= class_threshold <= 1:
        raise ValueError("invalid importance parameters")
    eligible = valid[:, :, None].expand(batch, frames, patch_count).clone()
    if motion_only:
        rgb = images.float()
        diff = torch.zeros_like(rgb[..., :1])
        diff[:, 1:] = (rgb[:, 1:] - rgb[:, :-1]).abs().mean(-1, keepdim=True) / 2
        motion = F.adaptive_avg_pool2d(
            diff.reshape(batch * frames, height, width, 1).permute(0, 3, 1, 2),
            (side, side)).reshape(batch, frames, patch_count)
        if current_rgb is not None:
            current = torch.as_tensor(current_rgb, device=images.device).float()
            transition = (current - rgb[:, -1]).abs().mean(-1, keepdim=True) / 2
            motion[:, -1] = torch.maximum(motion[:, -1],
                F.adaptive_avg_pool2d(transition.permute(0, 3, 1, 2), (side, side)).flatten(1))
        first = valid & (valid.long().cumsum(1) == 1)
        eligible &= (motion > threshold) | first[:, :, None]
    age = (valid.long().sum(1)[:, None] - valid.long().cumsum(1)).clamp_min(0).float()
    score = torch.exp(-(age[:, :, None] + (1 if reference_dino is not None else 0)) / time_scale).expand_as(eligible).float().clone()
    if contact_duration is not None:
        duration = torch.as_tensor(contact_duration, device=images.device).float().reshape_as(score)
        score += 1 - torch.exp(-duration.clamp_min(0) / contact_scale)
    if dino is not None:
        features = torch.as_tensor(dino, device=images.device).float()
        if features.ndim != 4 or features.shape[:3] != (batch, frames, patch_count):
            raise ValueError("DINO index must be [batch,frames,patches,features]")
        reference = None if reference_dino is None else torch.as_tensor(
            reference_dino, device=images.device).float()
        if reference is not None and reference.shape != (batch, patch_count, features.shape[-1]):
            raise ValueError("current DINO reference must be [batch,patches,features]")
        for i in range(batch):
            n = frames * patch_count
            base = features[i].reshape(n, -1)
            indexed = base if reference is None else torch.cat((base, reference[i]), 0)
            normalized = F.normalize(indexed, dim=-1)
            present = indexed.ne(0).any(-1)
            times = torch.arange(frames, device=images.device).repeat_interleave(patch_count)
            if reference is not None:
                times = torch.cat((times, times.new_full((patch_count,), frames)))
            latest = times[:n].clone()
            for start in range(0, n, 128):
                end = min(start + 128, n)
                match = (normalized[start:end] @ normalized.T >= class_threshold)
                match &= present[start:end, None] & present[None, :]
                latest[start:end] = torch.where(match, times[None, :], -1).amax(1).clamp_min(times[start:end])
            recency = torch.exp(-(latest - times[:n]).float() / class_recency_scale)
            score[i] += (recency * present[:n]).reshape(frames, patch_count)
    flat_score = score.flatten(1)
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


def select_jax(images, valid, patch_count, *, capacity=1024, top_k=1024,
               threshold=0.04, motion_only=False, dino=None,
               reference_dino=None, current_rgb=None, contact_duration=None,
               class_threshold=0.9, time_scale=8.0, contact_scale=8.0,
               class_recency_scale=8.0):
    """JAX equivalent; DINO is metadata used only for class recency."""
    import jax
    import jax.numpy as jnp

    _validate(capacity, top_k, patch_count)
    batch, frames, height, width, channels = images.shape
    side = math.isqrt(patch_count)
    if channels != 3 or side * side != patch_count or valid.shape != (batch, frames):
        raise ValueError("RGB memory images/masks do not match a square patch grid")
    if min(time_scale, contact_scale, class_recency_scale) <= 0 or not 0 <= class_threshold <= 1:
        raise ValueError("invalid importance parameters")
    eligible = jnp.broadcast_to(valid[:, :, None], (batch, frames, patch_count))
    if motion_only:
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
            transition = jnp.mean(jnp.abs(current_rgb - images[:, -1]), axis=-1, keepdims=True) / 2
            if height % side == 0 and width % side == 0:
                transition = transition.reshape(batch, side, height // side,
                                                side, width // side, 1).mean(axis=(2, 4))
            else:
                transition = jax.image.resize(transition, (batch, side, side, 1), method="linear")
            motion = motion.at[:, -1].set(jnp.maximum(motion[:, -1],
                                                      transition.reshape(batch, patch_count)))
        first = valid & (jnp.cumsum(valid.astype(jnp.int32), axis=1) == 1)
        eligible &= (motion > threshold) | first[:, :, None]
    age = jnp.maximum(jnp.sum(valid, axis=1)[:, None] -
                      jnp.cumsum(valid.astype(jnp.int32), axis=1), 0)
    score = jnp.broadcast_to(jnp.exp(-(age[:, :, None] + (1 if reference_dino is not None else 0)) / time_scale),
                             (batch, frames, patch_count))
    if contact_duration is not None:
        duration = jnp.asarray(contact_duration).reshape(score.shape)
        score += 1 - jnp.exp(-jnp.maximum(duration, 0) / contact_scale)
    if dino is not None:
        if dino.ndim != 4 or dino.shape[:3] != (batch, frames, patch_count):
            raise ValueError("DINO index must be [batch,frames,patches,features]")
        features = dino.reshape(batch, frames * patch_count, -1)
        if reference_dino is not None:
            if reference_dino.shape != (batch, patch_count, dino.shape[-1]):
                raise ValueError("current DINO reference must be [batch,patches,features]")
            indexed = jnp.concatenate((features, reference_dino), axis=1)
        else:
            indexed = features
        normalized = indexed / jnp.maximum(jnp.linalg.norm(indexed, axis=-1, keepdims=True), 1e-12)
        present = jnp.any(indexed != 0, axis=-1)
        times = jnp.repeat(jnp.arange(frames), patch_count)
        if reference_dino is not None:
            times = jnp.concatenate((times, jnp.full((patch_count,), frames)))
        # One block at a time avoids materializing an N x N similarity matrix.
        n = frames * patch_count
        block = 128
        pad = (-n) % block
        queries = jnp.pad(normalized[:, :n], ((0, 0), (0, pad), (0, 0)))
        q_present = jnp.pad(present[:, :n], ((0, 0), (0, pad)))
        q_times = jnp.pad(times[:n], (0, pad))
        def body(i, out):
            q = jax.lax.dynamic_slice(queries, (0, i * block, 0),
                                      (batch, block, queries.shape[-1]))
            q_ok = jax.lax.dynamic_slice(q_present, (0, i * block), (batch, block))
            q_t = jax.lax.dynamic_slice(q_times, (i * block,), (block,))
            sim = jnp.einsum("bqd,bkd->bqk", q, normalized)
            match = (sim >= class_threshold) & q_ok[:, :, None] & present[:, None, :]
            latest = jnp.max(jnp.where(match, times[None, None, :], -1), axis=-1)
            r = jnp.exp(-(jnp.maximum(latest, q_t[None, :]) - q_t[None, :]) / class_recency_scale) * q_ok
            return jax.lax.dynamic_update_slice(out, r, (0, i * block))
        values = jax.lax.fori_loop(0, (n + pad) // block, body,
                                  jnp.zeros((batch, n + pad), dtype=jnp.float32))
        score += values[:, :n].reshape(batch, frames, patch_count)
    flat = jnp.where(eligible, score, -jnp.inf).reshape(batch, -1)
    rank = jnp.argsort(jnp.argsort(-flat, axis=1, stable=True), axis=1)
    protected = (rank < top_k) & eligible.reshape(batch, -1)
    indices = jnp.arange(frames * patch_count, dtype=jnp.uint32)
    random = ((indices * jnp.uint32(1103515245) + jnp.uint32(12345))
              % jnp.uint32(2147483647)).astype(jnp.float32) / 2147483647
    priority = jnp.where(protected, 2 + flat, random[None, :])
    priority = jnp.where(eligible.reshape(batch, -1), priority, -jnp.inf)
    final_rank = jnp.argsort(jnp.argsort(-priority, axis=1, stable=True), axis=1)
    return ((final_rank < min(capacity, frames * patch_count)) &
            eligible.reshape(batch, -1)).reshape(batch, frames, patch_count)



def merge_single_frame_jax(old_metadata, current_index, generation, *, class_threshold=0.9):
    """Select N live positions from N prior plus N current SigLIP descriptors.

    Returns indices into [old,current] and updated (index,birth,latest). There is
    no evicted archive, so a discarded position cannot re-enter later.
    """
    import jax.numpy as jnp
    batch, n, _ = current_index.shape
    now = generation.astype(jnp.float32)
    if old_metadata is None:
        chosen = jnp.broadcast_to(jnp.arange(n)[None], (batch,n))
        birth = jnp.zeros((batch,n),jnp.float32)
        return chosen,current_index,birth,birth
    old_index,old_birth,old_latest=old_metadata
    similarity=jnp.einsum("bnd,bmd->bnm",old_index,current_index)
    matches=similarity>=class_threshold
    old_latest=jnp.where(jnp.any(matches,axis=-1),now,old_latest)
    repeated=jnp.sum(matches,axis=1).astype(jnp.float32)
    old_score=jnp.exp(-(now-old_birth)/8)+jnp.exp(-(old_latest-old_birth)/8)
    new_score=jnp.ones((batch,n),jnp.float32)+1/(1+repeated)
    chosen=jnp.argsort(-jnp.concatenate((old_score,new_score),axis=1),axis=1,stable=True)[:,:n]
    def gather(x):
        idx=chosen.reshape((batch,n)+((1,)*(x.ndim-2)))
        idx=jnp.broadcast_to(idx,(batch,n,*x.shape[2:]))
        return jnp.take_along_axis(x,idx,axis=1)
    new_birth=jnp.full((batch,n),now)
    return (chosen,gather(jnp.concatenate((old_index,current_index),axis=1)),
            gather(jnp.concatenate((old_birth,new_birth),axis=1)),
            gather(jnp.concatenate((old_latest,new_birth),axis=1)))

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
