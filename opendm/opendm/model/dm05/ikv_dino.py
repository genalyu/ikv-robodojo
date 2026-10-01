"""Frozen DINOv2 features used only as an RGB patch index.

The model is loaded from a local checkpoint on first use.  No feature is
projected into the policy's visual embeddings or attention K/V tensors.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F


@lru_cache(maxsize=4)
def _load_model(checkpoint: str, device: str):
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    )
    model = AutoModel.from_pretrained(
        checkpoint, local_files_only=True, trust_remote_code=False
    ).to(device)
    model.requires_grad_(False)
    model.eval()
    mean = tuple(float(x) for x in processor.image_mean)
    std = tuple(float(x) for x in processor.image_std)
    if len(mean) != 3 or len(std) != 3 or any(x <= 0 for x in std):
        raise ValueError("DINOv2 processor must provide three-channel mean/std")
    return model, mean, std


@torch.no_grad()
def encode_dino_grid(rgb_frames, *, checkpoint: str, grid: tuple[int, int], device="cpu"):
    """Encode chronological RGB frames as [T, grid_h, grid_w, D].

    Inputs may be PIL images, uint8 HWC arrays, or [T,3,H,W] tensors in
    [0,1]. The entire image is resized without a center crop so the index
    remains spatially aligned with the policy's image patches.
    """
    if not checkpoint:
        raise ValueError("a local DINOv2 checkpoint is required")
    if isinstance(rgb_frames, torch.Tensor):
        images = rgb_frames.detach()
        if images.ndim != 4:
            raise ValueError("DINO RGB tensor must be [T,3,H,W] or [T,H,W,3]")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
    else:
        images = torch.stack([
            torch.as_tensor(np.asarray(frame).copy()).permute(2, 0, 1)
            for frame in rgb_frames
        ])
    if images.ndim != 4 or images.shape[1] != 3 or not len(images):
        raise ValueError("DINO needs nonempty chronological RGB frames")
    images = images.to(device=device, dtype=torch.float32)
    if images.max() > 1:
        images = images / 255.0
    if images.min() < 0 or images.max() > 1 or not torch.isfinite(images).all():
        raise ValueError("DINO RGB values must be finite in [0,1]")
    images = F.interpolate(images, (224, 224), mode="bilinear", align_corners=False)
    model, channel_mean, channel_std = _load_model(str(checkpoint), str(device))
    mean = images.new_tensor(channel_mean).view(1, 3, 1, 1)
    std = images.new_tensor(channel_std).view(1, 3, 1, 1)
    patch = model.config.patch_size
    patch_h, patch_w = (patch, patch) if isinstance(patch, int) else tuple(patch)
    source_h, source_w = 224 // patch_h, 224 // patch_w
    pieces = []
    for batch in ((images - mean) / std).split(8):
        features = model(pixel_values=batch).last_hidden_state
        count = source_h * source_w
        if features.shape[1] < count:
            raise ValueError("DINO output has fewer tokens than its patch grid")
        patches = features[:, -count:].transpose(1, 2).reshape(
            len(batch), features.shape[-1], source_h, source_w
        )
        pooled = F.adaptive_avg_pool2d(patches.float(), grid)
        pieces.append(pooled.permute(0, 2, 3, 1).cpu())
    return torch.cat(pieces)
