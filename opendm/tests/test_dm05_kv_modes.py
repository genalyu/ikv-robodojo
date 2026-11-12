import pytest
import torch
from opendm.model.dm05.history_update import select_history_update
from opendm.model.dm05.kv_session import DM05KVSession
from opendm.model.dm05.persistent_kv import PersistentKVBank
from opendm.model.dm05.qwen_archive import QwenArchiveBank
from opendm.model.dm05.formula_archive import FormulaArchiveBank
from test_scorer_rl_reward import tiny


def step(start, end):
    ids = torch.arange(start * 16, end * 16)
    base = torch.tensor([[2] + [7] * 320 + [3]])
    return dict(input_ids=base, token_type_ids=torch.zeros_like(base), attention_mask=torch.ones_like(base),
                history_patch_ids=ids, history_times=(ids // 16).float(),
                history_features=torch.randn(len(ids), 32), history_descriptors=torch.ones(len(ids), 2),
                action_mask=torch.ones(1, 1, 4))


class Scorer:
    def __init__(self): self.times = []
    def score(self, candidates, current_time):
        self.times.append(current_time)
        return {c['id']: -int(c['id']) for c in candidates}


def archive_step(start, end, kind):
    data = step(start, end)
    if kind == 'qwen': data['history_descriptors'] = (data['history_patch_ids'] // 16).reshape(-1, 1)
    return data


@pytest.mark.parametrize('kind', ['persistent', 'qwen', 'formula'])
@pytest.mark.parametrize('mode', ['window_refresh', 'incremental'])
def test_real_dm05_kv_reuse_refresh_reset_and_no_duplicate_rows(kind, mode):
    model = tiny()
    factory = {'persistent': lambda: PersistentKVBank(320),
               'qwen': lambda: QwenArchiveBank(320, Scorer(), 'task', protect_current=False),
               'formula': lambda: FormulaArchiveBank(320)}[kind]
    session = DM05KVSession(model, update_mode=mode, bank_factory=factory)
    a = archive_step(0, 2, kind)
    out = session.inference_action(a, diffusion_steps=1)
    assert torch.isfinite(out).all()
    old = {i: (k.clone(), v.clone()) for i, (k, v) in session.bank.layers.items()}
    b = archive_step(0, 3, kind)  # Deliberately changed features for overlapping IDs.
    original_features = b['history_features'].clone()
    prepared = select_history_update(b, session.bank, mode)
    assert len(prepared['history_patch_ids']) == (16 if mode == 'incremental' else 48)
    session.inference_action(b, diffusion_steps=1)
    assert torch.equal(b['history_features'], original_features)
    assert session.bank.metadata['ids'].tolist() == list(range(48))
    for i, (k, v) in session.bank.layers.items():
        assert k.shape[-2] == 48
        if mode == 'incremental':
            assert torch.equal(k[..., :32, :], old[i][0])
            assert torch.equal(v[..., :32, :], old[i][1])
        else:
            assert not torch.equal(k[..., :32, :], old[i][0])
    # Same observed history can be used again without creating duplicate KV.
    session.inference_action(b, diffusion_steps=1)
    assert len(session.bank.metadata['ids']) == 48
    session.reset()
    assert session.bank.generation == 0 and session.bank.metadata is None
    assert len(select_history_update(b, session.bank, mode)['history_patch_ids']) == 48


def test_incremental_filters_before_visual_encoding_and_failed_forward_retry(monkeypatch):
    from opendm.model.dm05 import streaming_training as st
    model = tiny()
    session = DM05KVSession(model, update_mode='incremental')
    data = step(0, 1)
    session.inference_action(data, diffusion_steps=1)
    data = step(0, 3)
    data.pop('history_features')
    data['history_pixel_values'] = torch.randn(3, 3, 16, 16)
    counts = []
    def encode(model, pixels):
        counts.append(len(pixels))
        return torch.ones(len(pixels), 16, 32)
    monkeypatch.setattr(st, 'encode_policy_history', encode)
    original = model.inference_action
    def fail(**kwargs): raise RuntimeError('injected failure before commit')
    monkeypatch.setattr(model, 'inference_action', fail)
    with pytest.raises(RuntimeError, match='injected'): session.inference_action(data)
    assert session.bank.watermark == 15
    monkeypatch.setattr(model, 'inference_action', original)
    session.inference_action(data, diffusion_steps=1)
    assert counts == [2, 2]  # Backlog processed; failed attempt did not advance cursor.
    session.inference_action(data, diffusion_steps=1)
    assert counts == [2, 2]  # No new frame, no vision re-encoding.
    session.update_mode = 'window_refresh'
    session.inference_action(data, diffusion_steps=1)
    assert counts == [2, 2, 3]


def test_mode_validation_alignment_and_no_resurrection_of_evicted_frames():
    b = PersistentKVBank(32)
    b.watermark = 47
    b.metadata = {'ids': torch.arange(16, 48)}
    data = step(1, 3)
    assert len(select_history_update(data, b, 'incremental')['history_patch_ids']) == 0
    with pytest.raises(ValueError, match='mode'): select_history_update(data, b, 'typo')
    data['history_patch_ids'] = torch.arange(1, 33)
    with pytest.raises(ValueError, match='complete frames'): select_history_update(data, b, 'incremental')
    data = step(1, 3); data['history_times'][0] += .2
    with pytest.raises(ValueError, match='timestamps'): select_history_update(data, b)
    data = step(0, 1)
    with pytest.raises(ValueError, match='chronological'): select_history_update(data, b)


def test_archive_empty_updates_keep_query_clock_and_budget():
    scorer = Scorer()
    b = QwenArchiveBank(16, scorer, 'task', protect_current=False)
    for i in range(2):
        ids = torch.arange(i*16, (i+1)*16)
        p = b.plan(ids, (ids//16).reshape(-1, 1), torch.full((16,), float(i)))
        x = ids.float()[None, None, :, None]
        kv = p.merge(0, x, x, -2); b.commit(p, {0: kv}, 1)
    p = b.plan(torch.empty(0, dtype=torch.long), torch.empty(0, 1), torch.empty(0))
    assert scorer.times[-1] == 1.0 and len(p.metadata['ids']) == 16
    p = b.plan(torch.empty(0, dtype=torch.long), torch.empty(0, 1), torch.empty(0), query_time=5.0)
    assert scorer.times[-1] == 5.0
    with pytest.raises(ValueError, match='query_time'):
        b.plan(torch.empty(0, dtype=torch.long), torch.empty(0, 1), torch.empty(0), query_time=.5)


def test_incremental_training_uses_shared_mode_path():
    from opendm.model.dm05.streaming_training import train_episode
    model = tiny()
    steps = [step(0, 1), step(0, 2)]
    for data in steps:
        data['action'] = torch.randn(1, 2, 4)
        data['action_mask'] = torch.ones(1, 2, 4)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    before = model.model.action_out_proj.weight.detach().clone()
    loss = train_episode(model, steps, opt, update_mode='incremental')
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before, model.model.action_out_proj.weight)


@pytest.mark.parametrize('kind', ['persistent', 'qwen', 'formula'])
def test_empty_initial_history_and_missing_descriptors_on_repeated_window(kind):
    factories = {'persistent': lambda: PersistentKVBank(320),
                 'qwen': lambda: QwenArchiveBank(320, Scorer(), 'task', protect_current=False),
                 'formula': lambda: FormulaArchiveBank(320)}
    session = DM05KVSession(tiny(), update_mode='incremental', bank_factory=factories[kind])
    empty = step(0, 0)
    empty.pop('history_descriptors')
    session.inference_action(empty, diffusion_steps=1)
    data = archive_step(0, 2, kind)
    session.inference_action(data, diffusion_steps=1)
    data.pop('history_descriptors')
    session.inference_action(data, diffusion_steps=1)
    assert session.bank.metadata['ids'].tolist() == list(range(32))


@pytest.mark.parametrize('kind', ['qwen', 'formula'])
def test_archive_requires_explicit_descriptor_contract(kind):
    factory = (lambda: QwenArchiveBank(320, Scorer(), 'task')) if kind == 'qwen' else (lambda: FormulaArchiveBank(320))
    session = DM05KVSession(tiny(), bank_factory=factory)
    data = step(0, 1)
    data.pop('history_descriptors')
    with pytest.raises(ValueError, match='explicit history_descriptors'):
        session.inference_action(data, diffusion_steps=1)
    assert session.bank.generation == 0


def test_incremental_archive_recall_uses_saved_kv_after_budget_overflow():
    class ToggleScorer:
        def __init__(self): self.prefer_new = True
        def score(self, candidates, current_time):
            sign = 1 if self.prefer_new else -1
            return {c['id']: sign * int(c['id']) for c in candidates}
    scorer = ToggleScorer()
    session = DM05KVSession(tiny(), update_mode='incremental',
                            bank_factory=lambda: QwenArchiveBank(320, scorer, 'task', protect_current=False))
    session.inference_action(archive_step(0, 20, 'qwen'), diffusion_steps=1)
    original = {i: (k[..., :16, :].clone(), v[..., :16, :].clone())
                for i, (k, v) in session.bank.layers.items()}
    data = archive_step(1, 21, 'qwen')
    session.inference_action(data, diffusion_steps=1)
    assert session.bank.metadata['ids'].tolist() == list(range(16, 336))
    assert len(session.bank.archive_metadata['ids']) == 336
    scorer.prefer_new = False
    session.inference_action(data, diffusion_steps=1)  # Zero fresh rows, still score all candidates.
    assert session.bank.metadata['ids'].tolist() == list(range(320))
    for i, (k, v) in session.bank.layers.items():
        assert torch.equal(k[..., :16, :], original[i][0])
        assert torch.equal(v[..., :16, :], original[i][1])
