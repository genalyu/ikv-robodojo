"""Small CPU contract tests that do not load the WAN checkpoint."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys
import types

import numpy as np
from PIL import Image
import torch
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_motion_patch_selection_and_shared_video_key_gate():
    selector = _module("ikv_rgb", ROOT / "openwam/model/ikv_rgb.py")
    masks = _module("mask_modes", ROOT / "openwam/model/architectures/utils/mask_modes.py")
    frames = [Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8)) for _ in range(9)]
    frames[1] = Image.fromarray(np.full((32, 32, 3), 255, dtype=np.uint8))
    kept = selector.select_clean_prefix_patches(
        frames, clean_latent_frames=3, grid_height=4, grid_width=4,
        capacity=32, top_k=8, motion_threshold=0.04, motion_only=True, device="cpu",
    )
    assert kept.shape == (48,)
    assert kept.sum() == 32
    assert kept[:16].all()
    assert kept[16:32].all()
    assert not kept[-16:].any()
    video_keep = torch.cat([kept, torch.ones(16, dtype=torch.bool)])[None]
    state = SimpleNamespace(extras={"ikv_video_key_mask": video_keep})
    joint = masks.apply_ikv_video_key_mask(torch.ones(70, 70, dtype=torch.bool), state)
    assert joint.shape == (70, 70)
    assert torch.equal(joint[:, :64], video_keep.expand(70, -1))
    assert joint[:, 64:].all()
    keys = torch.arange(70).view(1, 70, 1, 1).float()
    values = keys + 100
    compact_keys, compact_values, compact_mask = masks.compact_ikv_joint_kv(
        keys, values, joint, state
    )
    kept_positions = torch.cat([video_keep[0].nonzero().flatten(), torch.arange(64, 70)])
    assert torch.equal(compact_keys.flatten(), kept_positions.float())
    assert torch.equal(compact_values.flatten(), kept_positions.float() + 100)
    assert torch.equal(compact_mask, joint.index_select(1, kept_positions))


def test_causal_vae_endpoints_require_latest_observation_alignment():
    selector = _module("ikv_rgb_alignment", ROOT / "openwam/model/ikv_rgb.py")
    frame = Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="causal VAE endpoints"):
        selector.select_clean_prefix_patches(
            [frame] * 8, clean_latent_frames=3, grid_height=2, grid_width=2,
            capacity=8, top_k=2, motion_threshold=0.04, device="cpu",
        )


def test_dino_grid_is_frozen_index_metadata(monkeypatch):
    dino = _module("ikv_dino_grid", ROOT / "openwam/model/ikv_dino.py")

    class FakeModel(torch.nn.Module):
        config = SimpleNamespace(patch_size=14)

        def forward(self, pixel_values):
            batch = pixel_values.shape[0]
            tokens = torch.ones(batch, 257, 3, device=pixel_values.device)
            return SimpleNamespace(last_hidden_state=tokens)

    monkeypatch.setattr(dino, "_load_model", lambda *_: (
        FakeModel(), (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    ))
    frame = np.zeros((32, 32, 3), dtype=np.uint8)
    grid = dino.encode_dino_grid([frame], checkpoint="local-only", grid=(2, 2))
    assert grid.shape == (1, 2, 2, 3)
    assert torch.equal(grid, torch.ones_like(grid))


def test_policy_keeps_single_current_frame_and_persistent_bank(monkeypatch):
    root = types.ModuleType("openwam")
    root.__path__ = []
    deploy = types.ModuleType("openwam.deploy")
    deploy.__path__ = []
    engine_module = types.ModuleType("openwam.deploy.engine")
    engine_module.BaseInferenceEngine = object
    executors = types.ModuleType("openwam.deploy.executors")

    class Executor:
        def __init__(self, **kwargs):
            pass

    executors.SyncInferenceExecutor = Executor
    executors.AsyncInferenceExecutor = Executor
    executors.normalize_execution_config = lambda _: SimpleNamespace(
        enabled=False, inference_horizon=1
    )
    for name, module in (
        ("openwam", root), ("openwam.deploy", deploy),
        ("openwam.deploy.engine", engine_module),
        ("openwam.deploy.executors", executors),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    policy_module = _module("ikv_policy_test", ROOT / "openwam/deploy/policy.py")
    cfg = SimpleNamespace(inference=SimpleNamespace(
        ikv_rgb_enabled=True, ikv_history_frames=16,
        ikv_require_dino=False,
    ))
    architecture = SimpleNamespace(video_backbone=SimpleNamespace(
        _is_ti2v=True, _temporal_compression=4
    ))
    policy = policy_module.WAMPolicy(SimpleNamespace(architecture=architecture), cfg)
    first = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    second = Image.fromarray(np.ones((8, 8, 3), dtype=np.uint8))
    policy._build_conditions({"image": first})
    conditions = policy._build_conditions({"image": second})
    assert conditions["first_frame_image"] == [second]
    assert "ikv_rgb_images" not in conditions
    assert conditions["ikv_kv_state"] is policy._ikv_kv_state


def test_importance_retains_static_history_without_motion_gate():
    selector = _module("ikv_rgb_static", ROOT / "openwam/model/ikv_rgb.py")
    frames = [Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)) for _ in range(9)]
    keep = selector.select_clean_prefix_patches(
        frames, clean_latent_frames=3, grid_height=2, grid_width=2,
        capacity=4, top_k=4, motion_threshold=0.04, device="cpu",
    )
    assert keep[-4:].all()
    motion = selector.select_clean_prefix_patches(
        frames, clean_latent_frames=3, grid_height=2, grid_width=2,
        capacity=4, top_k=4, motion_threshold=0.04,
        motion_only=True, device="cpu",
    )
    assert motion[:4].all()


def test_dino_class_recency_preserves_unique_older_patch():
    selector = _module("ikv_rgb_class", ROOT / "openwam/model/ikv_rgb.py")
    frames = [Image.fromarray(np.zeros((16, 16, 3), dtype=np.uint8)) for _ in range(9)]
    dino = torch.zeros(3, 4, 2)
    dino[0, 0, 0] = 1
    dino[0, 1, 1] = 1
    dino[2, 2, 1] = 1
    keep = selector.select_clean_prefix_patches(
        frames, clean_latent_frames=3, grid_height=2, grid_width=2,
        capacity=2, top_k=2, motion_threshold=0.04, device="cpu",
        dino_features=dino,
    ).reshape(3, 4)
    assert keep[0, 0] and keep[2, 2]
    assert not keep[0, 1]
