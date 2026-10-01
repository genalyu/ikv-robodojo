from types import SimpleNamespace

import torch
import numpy as np
from PIL import Image
import pytest

from opendm.model.dm05.dm05_arch import DM05ForConditionalGeneration
from opendm.model.dm05.ikv_rgb import (
    history_motion_scores,
    mask_history_prefix,
    select_history_patches,
)


def test_rgb_motion_patch_and_memory_anchors():
    frames = torch.zeros(4, 3, 32, 32)
    frames[1:, :, :8, :8] = 1
    scores = history_motion_scores(frames)
    assert scores.shape == (4, 16)
    assert scores[1, 0] > 0
    assert torch.count_nonzero(scores[1, 1:]) == 0
    selected = select_history_patches(frames, capacity=40, top_k=8, seed=9)
    assert selected.sum() == 17  # Only dense seed plus changed RGB patch.
    assert selected[0].all()
    assert not selected[-1].any()
    assert selected[1, 0]
    assert torch.equal(selected, select_history_patches(frames, capacity=40, top_k=8, seed=9))


def test_prefix_mask_preserves_feature_scatter_layout():
    frames = torch.zeros(2, 3, 32, 32)
    ids = torch.full((1, 36), 6, dtype=torch.long)
    history = torch.zeros_like(ids, dtype=torch.bool)
    history[:, 2:34] = True
    masked = mask_history_prefix(ids, history, frames, history_rgb_values=frames, capacity=16)
    assert masked.shape == ids.shape
    assert history.sum() == 32  # Vision scatter still consumes all 32 features.
    assert (masked[:, 2:18] == 6).all()
    assert (masked[:, 18:34] == 7).all()


def test_motion_reads_raw_rgb_not_normalized_vision_pixels():
    raw = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
    raw[1, :, :8, :8] = 255
    normalized_pixels = torch.zeros(2, 3, 32, 32)
    ids = torch.full((1, 32), 6, dtype=torch.long)
    history = torch.ones_like(ids, dtype=torch.bool)
    masked = mask_history_prefix(
        ids, history, normalized_pixels, history_rgb_values=raw,
        capacity=32, motion_threshold=0.04,
    )
    assert masked[0, 16] == 6
    assert (masked[0, 17:] == 7).all()


def test_last_history_patch_tracks_current_rgb_change():
    history = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
    current = torch.zeros(1, 3, 32, 32, dtype=torch.uint8)
    current[:, :, :8, :8] = 255
    scores = history_motion_scores(history, current_rgb=current)
    assert scores[-1, 0] == 1
    assert not scores[-1, 1:].any()
    selected = select_history_patches(history, current_rgb=current, capacity=32)
    assert selected[0].all() and selected[-1, 0]
    assert selected.sum() == 17


def test_history_and_head_share_training_augmentation():
    pytest.importorskip("albumentations")
    from opendm.data.augmentations import TrainingTransformPipeline
    from opendm.data.transforms import PixelTransform

    image = Image.fromarray(np.full((32, 48, 3), 127, dtype=np.uint8))
    transform = PixelTransform(
        TrainingTransformPipeline(p=1.0), consistent_history=True
    )
    data = transform({"images": [image], "history_images": [image, image]})
    assert np.array_equal(np.asarray(data["images"][0]),
                          np.asarray(data["history_images"][0]))
    assert np.array_equal(np.asarray(data["history_images"][0]),
                          np.asarray(data["history_images"][1]))


def test_dino_similarity_is_a_separate_retention_index():
    frames = torch.stack([torch.zeros(3, 8, 8), torch.ones(3, 8, 8), torch.zeros(3, 8, 8)])
    features = torch.zeros(3, 16, 2)
    features[-1, :, 0] = 1
    features[1, 3, 0] = 1
    features[1, :, 1] = 1
    selected = select_history_patches(
        frames, capacity=17, top_k=17, dino_features=features,
        reference_dino=torch.tensor([[1.0, 0.0]]).expand(16, -1),
        motion_threshold=0.01, seed=1,
    )
    assert selected.shape == (3, 16)
    assert selected.sum() == 17
    assert selected[-1].any()
    assert selected[1, 3]


def test_prefix_kv_compaction_uses_same_positions_in_every_layer():
    ids = torch.tensor([[2, 7, 5, 7, 6]])
    cache = SimpleNamespace(layers=[
        SimpleNamespace(keys=torch.arange(5).view(1, 1, 5, 1).clone(),
                        values=torch.arange(5).view(1, 1, 5, 1).clone())
        for _ in range(3)
    ])
    hidden = torch.arange(5).view(1, 5, 1)
    cache, hidden, ids = DM05ForConditionalGeneration._compact_ikv_cache(
        cache, hidden, ids, torch.ones(1, 5)
    )
    assert ids.tolist() == [[2, 5, 6]]
    assert hidden.flatten().tolist() == [0, 2, 4]
    for layer in cache.layers:
        assert layer.keys.flatten().tolist() == [0, 2, 4]
        assert layer.values.flatten().tolist() == [0, 2, 4]
