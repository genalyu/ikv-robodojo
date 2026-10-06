import torch
import pytest
import numpy as np
import jax.numpy as jnp
import importlib.util
from pathlib import Path
import sys
import types

from openpi.models.ikv_rgb import compact_torch_prefix_tokens, merge_single_frame_jax, select_torch, select_jax


def test_rgb_memory_patch_budget_and_motion():
    images = torch.zeros(1, 4, 32, 32, 3)
    images[:, 1:, :16, :16] = 1
    valid = torch.ones(1, 4, dtype=torch.bool)
    selected = select_torch(images, valid, 4, capacity=9, top_k=4, threshold=0.04, motion_only=True)
    assert selected.shape == (1, 4, 4)
    assert selected.sum() == 5
    assert selected[0, 0].all()
    assert not selected[0, -1].any()
    assert selected[0, 1, 0]


def test_jax_and_torch_select_same_changed_patches():
    pytest.importorskip("jax")
    import jax.numpy as jnp

    images = torch.zeros(1, 4, 32, 32, 3)
    images[:, 1:, :16, :16] = 1
    valid = torch.ones(1, 4, dtype=torch.bool)
    expected = select_torch(images, valid, 4, capacity=9, top_k=4)
    actual = select_jax(jnp.asarray(images.numpy()), jnp.asarray(valid.numpy()),
                        4, capacity=9, top_k=4)
    assert np.array_equal(np.asarray(actual), expected.numpy())


def test_latest_actual_rgb_marks_last_history_patch():
    history = torch.zeros(1, 2, 32, 32, 3)
    current = torch.zeros(1, 32, 32, 3)
    current[:, :16, :16] = 1
    selected = select_torch(
        history, torch.ones(1, 2, dtype=torch.bool), 4,
        current_rgb=current, capacity=8, motion_only=True,
    )
    assert selected[0, 0].all()
    assert selected[0, 1, 0]
    assert selected.sum() == 5


def test_selected_memory_tokens_are_gathered_before_prefill():
    tokens = torch.arange(9).float().reshape(1, 9, 1)
    mask = torch.tensor([[True, True, True, True, False, True, False, True, True]])
    gathered, visible = compact_torch_prefix_tokens(tokens, mask, 2, 5, 3)
    assert gathered.flatten().tolist() == [0, 1, 2, 3, 5, 7, 8]
    assert visible.all()


def test_robodojo_adapter_accepts_hwc_and_right_aligns_history(monkeypatch):
    # The input adapter itself is pure NumPy; keep this contract test runnable
    # on lightweight CPU installs without the full JAX/Flax training stack.
    fake_transforms = types.ModuleType("openpi.transforms")
    fake_transforms.DataTransformFn = type("DataTransformFn", (), {})
    monkeypatch.setitem(sys.modules, "openpi.transforms", fake_transforms)
    spec = importlib.util.spec_from_file_location(
        "aloha_policy_ikv_test", Path(__file__).resolve().parents[1] / "src/openpi/policies/aloha_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    data = {
        "state": np.zeros(14, dtype=np.float32),
        "images": {"cam_high": image},
        "history_images": [image, np.full_like(image, 255)],
    }
    result = module.AlohaIKVRGBInputs(adapt_to_pi=False, max_history_images=3)(data)
    assert result["image_mask"]["memory_000_rgb"] == np.False_
    assert result["image_mask"]["memory_001_rgb"] == np.True_
    assert result["image_mask"]["memory_002_rgb"] == np.True_
    assert result["image"]["memory_000_rgb"].shape == image.shape
    assert result["image"]["memory_002_rgb"].mean() == 255


def test_precomputed_dino_index_follows_right_aligned_history(monkeypatch):
    fake_transforms = types.ModuleType("openpi.transforms")
    fake_transforms.DataTransformFn = type("DataTransformFn", (), {})
    monkeypatch.setitem(sys.modules, "openpi.transforms", fake_transforms)
    spec = importlib.util.spec_from_file_location(
        "aloha_policy_dino_test", Path(__file__).resolve().parents[1] / "src/openpi/policies/aloha_policy.py"
    )
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    features = np.ones((2, 256, 3), dtype=np.float32)
    result = module.AlohaIKVRGBInputs(max_history_images=3, ikv_require_dino=True)({
        "state": np.zeros(14, dtype=np.float32), "images": {"cam_high": image},
        "history_images": [image, image], "ikv_dino_features": features,
        "ikv_reference_dino": np.ones((256, 3), dtype=np.float32),
    })
    assert result["ikv_dino_features"].shape == (3, 256, 3)
    assert not result["ikv_dino_features"][0].any()
    assert result["ikv_dino_features"][1:].all()
    assert result["ikv_reference_dino"].shape == (256, 3)


def test_importance_and_motion_gates_are_independent():
    images = torch.zeros(1, 3, 32, 32, 3)
    valid = torch.ones(1, 3, dtype=torch.bool)
    selected = select_torch(images, valid, 4, capacity=4, top_k=4)
    assert selected[0, -1].all()
    motion = select_torch(images, valid, 4, capacity=4, top_k=4, motion_only=True)
    assert motion[0, 0].all()


def test_dino_class_recency_matches_jax_and_torch():
    import jax.numpy as jnp
    import numpy as np
    images = np.zeros((1, 3, 32, 32, 3), dtype=np.float32)
    valid = np.ones((1, 3), dtype=bool)
    dino = np.zeros((1, 3, 4, 2), dtype=np.float32)
    dino[0, 0, 0, 0] = 1
    dino[0, 0, 1, 1] = 1
    dino[0, 2, 2, 1] = 1
    torch_keep = select_torch(torch.tensor(images), torch.tensor(valid), 4,
                              capacity=2, top_k=2, dino=torch.tensor(dino))
    jax_keep = select_jax(jnp.array(images), jnp.array(valid), 4,
                          capacity=2, top_k=2, dino=jnp.array(dino))
    assert np.array_equal(torch_keep.numpy(), np.asarray(jax_keep))
    assert bool(torch_keep[0, 0, 0]) and bool(torch_keep[0, 2, 2])


def test_single_frame_jax_fixed_budget_history_and_no_archive():
    n=8
    current=jnp.zeros((1,n,2)); current=current.at[...,1].set(1); current=current.at[0,0].set(jnp.array([1.,0.]))
    chosen,index,birth,latest=merge_single_frame_jax(None,current,jnp.array(0))
    assert chosen.shape==(1,n) and np.array_equal(np.asarray(chosen),np.arange(n)[None])
    # Repeated background is discounted; the unique old position survives.
    next_index=jnp.zeros_like(current).at[...,1].set(1)
    chosen2,index2,birth2,latest2=merge_single_frame_jax((index,birth,latest),next_index,jnp.array(1))
    assert 0 in np.asarray(chosen2)[0]
    assert chosen2.shape==(1,n)
    # State contains exactly live metadata; no archive/tombstone output exists.
    assert index2.shape==(1,n,2) and birth2.shape==latest2.shape==(1,n)
