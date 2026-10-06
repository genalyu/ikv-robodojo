from types import SimpleNamespace
import torch
import pytest
from opendm.model.dm05.online_history import (
    OnlineHistoryMemory, replay_history_indices, pack_history_features, prepare_training_memory,
)


def observation(t):
    ids = torch.arange(t * 16, (t + 1) * 16).reshape(16, 1)
    dino = torch.zeros(16, 2)
    dino[:, 1] = 1
    if t == 0:
        dino[0] = torch.tensor([1., 0.])
    return ids, dino, torch.zeros(1, 3, 8, 8)


def test_irreversible_bounded_memory_can_keep_very_old_event():
    bank = OnlineHistoryMemory()
    evicted = set()
    previous = set()
    for t in range(65):
        ids, dino, rgb = observation(t)
        candidates = previous | set(ids.flatten().tolist())
        bank.update(ids, dino, t, rgb)
        retained = set(bank.read().flatten().tolist())
        assert len(retained) == min((t + 1) * 16, 320)
        assert retained <= candidates
        assert not retained & evicted
        evicted |= candidates - retained
        previous = retained
    assert 0 in retained  # A unique event survives well beyond the last 20 frames.
    assert 1 not in retained
    assert bank.features.shape[0] == bank.dino.shape[0] == 320
    assert bank.previous_rgb.shape[0] == 1


def test_training_replay_matches_online_and_motion_only_is_read_gate():
    frames = [observation(t) for t in range(30)]
    bank = OnlineHistoryMemory()
    gated = OnlineHistoryMemory(motion_only=True)
    for t, (ids, dino, rgb) in enumerate(frames):
        bank.update(ids, dino, t, rgb)
        gated.update(ids, dino, t, rgb)
    rgb = torch.cat([x[2] for x in frames])
    dino = torch.stack([x[1] for x in frames])
    replayed = replay_history_indices(rgb, dino)
    assert torch.equal(replayed, bank.read()[:, 0])
    assert torch.equal(gated.features, bank.features)
    assert len(gated.read()) < len(bank.read())
    with pytest.raises(ValueError, match="strictly increase"):
        bank.update(*frames[-1][:2], 29, frames[-1][2])


def test_pack_arbitrary_patch_count_and_gradients():
    ids = torch.tensor([[2] + [7] * 320 + [5], [2] + [7] * 320 + [5]])
    values = torch.randn(37, 4, requires_grad=True)
    packed, types, mask, features = pack_history_features(ids, torch.zeros_like(ids),
                                                         [values, values[:0]])
    assert mask.sum(1).tolist() == [37, 0]
    assert packed[0, 0] == 2 and packed[0, -1] == 5
    assert torch.all(types[mask] == 1)
    features.sum().backward()
    assert torch.all(values.grad == 1)


def test_training_replay_batch_offsets_and_visual_gradient(monkeypatch):
    import opendm.model.dm05.online_history as online
    import opendm.model.dm05.ikv_dino as dino_module
    model = torch.nn.Linear(1, 1, bias=False)
    model.config = SimpleNamespace(ikv_dino_model_path="fake", ikv_history_capacity=320,
                                   ikv_motion_only=False, ikv_motion_threshold=0.04)
    model.model = SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace(hidden_size=1)))
    monkeypatch.setattr(dino_module, "encode_dino_grid",
                        lambda rgb, **kw: torch.ones(len(rgb), 4, 4, 2))
    monkeypatch.setattr(online, "encode_policy_history",
                        lambda model, pixels: model(pixels[:, :1]).view(-1, 1, 1).expand(-1, 16, 1))
    pixels = torch.arange(1., 26.).reshape(25, 1)
    rgb = torch.zeros(25, 3, 8, 8)
    features = prepare_training_memory(model, pixels, rgb, torch.tensor([24, 0, 1]))
    assert [len(x) for x in features] == [320, 0, 16]
    assert torch.equal(features[2], model(pixels[-1:]).expand(16, 1))
    sum(x.sum() for x in features).backward()
    assert model.weight.grad.item() > 0


def test_training_load_history_uses_episode_anchored_sampling(monkeypatch):
    import orjson
    import opendm.data.transforms as transforms
    monkeypatch.setattr(transforms, "_load_image", lambda path: path)
    data = {"meta_data": {"fps": 25, "frame_index": 526},
            "raw_lines": [orjson.dumps({"images_1": {"type": "image", "url": str(i)}})
                          for i in range(527)]}
    result = transforms.LoadHistory(max_history_images=None)(data)
    assert result["history_images"] == [str(i) for i in range(0, 502, 25)]

def test_adapter_refreshes_features_and_isolates_resets(monkeypatch):
    import sys
    import numpy as np
    from collections import deque
    sys.path.insert(0, "/home/ubuntu/genalyu/RoboDojo")
    from XPolicyLab.policy.OpenDM.model import Model
    import opendm.exp.dm05_exp as exp
    import opendm.model.dm05.online_history as online
    import opendm.model.dm05.ikv_dino as dino_module
    model = torch.nn.Linear(1, 1)
    model.config = SimpleNamespace(ikv_history_capacity=320, ikv_motion_only=False,
                                   ikv_motion_threshold=0.04, ikv_dino_model_path="fake")
    model.precision_policy = "bf16_mixed"
    model.model = SimpleNamespace(language_model=SimpleNamespace(config=SimpleNamespace(hidden_size=1)))
    monkeypatch.setattr(exp, "unwrap_dm05_model", lambda x: x)
    calls = []
    def encode(model, pixels):
        calls.append(1)
        return torch.ones(1, 16, 1)
    monkeypatch.setattr(online, "encode_policy_history", encode)
    monkeypatch.setattr(dino_module, "encode_dino_grid", lambda *args, **kwargs: torch.ones(1, 4, 4, 2))
    owner = object.__new__(Model)
    owner._inference = SimpleNamespace(model=model, device="cpu", processor=SimpleNamespace(
        image_processor=lambda **kw: {"pixel_values": torch.zeros(1, 3, 8, 8)}))
    owner.history_slots = 20
    owner.history_action_interval = 25
    owner._ikv_memory_by_env = {}
    key = owner._state_key(0, "episode-a")
    other = owner._state_key(0, "episode-b")
    owner._history_by_env = {key: deque([(1, 0, np.zeros((8, 8, 3), dtype=np.uint8))])}
    owner._history_sequence_by_env = {key: 26}
    owner._observations = {}
    owner._latest_env_idx_by_evaluation = {}
    owner._latest_env_idx_list = [0]
    assert len(owner._online_history_payload_for(0, "episode-a")["history_features"]) == 16
    assert len(owner._online_history_payload_for(0, "episode-a")["history_features"]) == 16
    assert len(calls) == 2  # Window features refresh; raw window remains bounded.
    assert len(owner._history_by_env[key]) == 1
    owner._ikv_memory_by_env[other] = object()
    owner._clear_evaluation_state("episode-a")
    assert key not in owner._ikv_memory_by_env and other in owner._ikv_memory_by_env
    owner.reset()
    assert not owner._ikv_memory_by_env


def test_training_tokenizer_retains_candidates_but_fixes_context_budget():
    import numpy as np
    from PIL import Image
    from opendm.data.transforms import ChatTokenization
    class Processor:
        tokenizer = SimpleNamespace(convert_tokens_to_ids=lambda s: 6)
        image_processor = staticmethod(lambda images, **kw: {"pixel_values": torch.zeros(len(images), 3, 8, 8)})
        def apply_chat_template(self, messages, **kw):
            text = messages[0]["content"][0]["text"]
            ids = torch.tensor([[7] * text.count("<unused1>") + [6] * text.count("<unused0>") + [2]])
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids),
                    "token_type_ids": torch.zeros_like(ids), "pixel_values": torch.zeros(0, 3, 8, 8)}
    transform = ChatTokenization(Processor(), max_length=1536, image_prompts=[], add_state=False,
                                 is_history=True, include_history_rgb=True, max_history_images=20,
                                 online_history=True)
    frame = Image.fromarray(np.zeros((8, 8, 3), dtype=np.uint8))
    sample = transform({"history_images": [frame] * 35, "images": [], "prompt": "test"})
    assert sample["history_frame_counts"].item() == 35
    assert len(sample["history_pixel_values"]) == 35
    assert (sample["input_ids"] == 7).sum() == 320
    assert sample["history_mask"].sum() == 0
    assert sample["input_ids"].shape[1] == 321

def test_real_tiny_dm05_training_replay_backward_and_inference(monkeypatch):
    from transformers import Gemma3Config, Gemma3TextConfig, SiglipVisionConfig
    from opendm.model.dm05.dm05_arch import DM05Config, DM05ForConditionalGeneration
    import opendm.model.dm05.ikv_dino as dino_module
    text = Gemma3TextConfig(vocab_size=128, hidden_size=32, intermediate_size=64,
                           num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                           head_dim=8, layer_types=["full_attention", "full_attention"],
                           max_position_embeddings=1024)
    vision = SiglipVisionConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                                num_attention_heads=4, image_size=16, patch_size=4)
    vlm = Gemma3Config(text_config=text.to_dict(), vision_config=vision.to_dict(),
                       mm_tokens_per_image=16, image_token_index=127)
    config = DM05Config(vlm_config=vlm, action_config=text, action_dim=4, chunk_size=2,
                        ikv_rgb_enabled=True, ikv_dino_model_path="fake")
    model = DM05ForConditionalGeneration(config)
    # Gemma initializes its multimodal projection to zero; emulate a trained checkpoint.
    torch.nn.init.normal_(model.model.vlm.model.multi_modal_projector.mm_input_projection_weight, std=0.02)
    monkeypatch.setattr(dino_module, "encode_dino_grid",
                        lambda rgb, **kw: torch.ones(len(rgb), 4, 4, 2))
    ids = torch.tensor([[2] + [7] * 320 + [3]])
    model.train()
    out = model(input_ids=ids, attention_mask=torch.ones_like(ids),
                token_type_ids=torch.zeros_like(ids), history_mask=torch.zeros_like(ids, dtype=torch.bool),
                history_frame_counts=torch.tensor([22]), history_pixel_values=torch.randn(22, 3, 16, 16),
                history_rgb_values=torch.zeros(22, 3, 16, 16),
                action=torch.randn(1, 2, 4), action_mask=torch.ones(1, 2, 4))
    assert torch.isfinite(out.loss)
    out.loss.backward()
    grads = [p.grad for p in model.model.vlm.model.vision_tower.parameters() if p.grad is not None]
    assert grads and any(g.abs().sum() > 0 for g in grads)
    model.eval()
    features = torch.randn(320, 32)
    ids, types, mask, features = pack_history_features(ids, torch.zeros_like(ids), [features])
    with torch.no_grad():
        actions = model.inference_action(input_ids=ids, attention_mask=torch.ones_like(ids),
                                         token_type_ids=types, history_mask=mask, history_features=features,
                                         diffusion_steps=2, action_mask=torch.ones(1, 1, 4))
    assert actions.shape == (1, 2, 4) and torch.isfinite(actions).all()
