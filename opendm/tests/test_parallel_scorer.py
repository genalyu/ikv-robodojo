from types import SimpleNamespace
import pytest
import torch
from opendm.model.dm05.parallel_scorer import ParallelScorerClient
from opendm.model.dm05.qwen_archive import QwenArchiveBank
from opendm.model.dm05.formula_archive import FormulaArchiveBank


class Encoder:
    revision = 'test-encoder'
    training = False
    def task(self, text):
        return torch.ones(1, 3)
    def images(self, images):
        return torch.cat(images)


class Scorer:
    info = {'encoder_revision': 'test-encoder'}
    model = SimpleNamespace(training=False)
    def score(self, packet, budgets):
        self.packet = packet
        return {'scores': packet['features'][:, 0]}


def client():
    return ParallelScorerClient(Scorer(), Encoder(), 'task')


def test_mapping_causality_and_current_context():
    c = client()
    c.register(25, 1., torch.arange(64.).reshape(64, 1))
    c.register(50, 2., torch.ones(64, 1))
    c.register(75, 3., torch.ones(64, 1)*999)
    result = c.score([dict(id='old', frame=25, patch=0, time=1.),
                      dict(id='last', frame=25, patch=15)], 2.)
    assert result == {'old': 4.5, 'last': 58.5}
    assert len(c.scorer.packet['features']) == 128
    assert c.scorer.packet['query_time'] == 2.
    with pytest.raises(ValueError, match='future'):
        c.score([dict(id='future', frame=75, patch=0)], 2.)


def test_identity_and_revision_validation():
    c = client(); x = torch.ones(64, 2)
    c.register(0, 0, x); c.register(0, 0, x)
    with pytest.raises(ValueError, match='identity'):
        c.register(0, 0, x+1)
    with pytest.raises(ValueError, match='timestamp'):
        c.append([0], [x], [2])
    with pytest.raises(ValueError, match='time mismatch'):
        c.score([dict(id='a', frame=0, patch=0, time=2)], 3)
    e = Encoder(); e.revision = 'different'
    with pytest.raises(ValueError, match='revision'):
        ParallelScorerClient(Scorer(), e, 'task')


def test_local_scorer_drives_archive_selection_and_exact_payloads():
    c = client(); c.append([0, 25], [torch.ones(64, 2)*10, torch.zeros(64, 2)], [0., 1.])
    b = QwenArchiveBank(16, c, 'task', protect_current=False)
    ids = torch.arange(32)
    p = b.plan(ids, torch.tensor([0]*16+[25]*16), torch.tensor([0.]*16+[1.]*16))
    assert p.metadata['ids'].tolist() == list(range(16))
    payload = ids.float()[None, None, :, None]
    k, v = p.merge(0, payload, payload+100, -2)
    assert k.flatten().tolist() == list(range(16))
    assert v.flatten().tolist() == list(range(100,116))
    b.commit(p, {0: (k,v)}, 1)
    assert len(b.archive_metadata['ids']) == 32


def test_formula_plan_does_not_mutate_content_before_commit():
    b = FormulaArchiveBank(2, weights=(0,0,0,1))
    ids = torch.arange(4)
    descriptors = torch.tensor([[1.,0.,.1],[0.,1.,.8],[1.,0.,.4],[0.,1.,.9]])
    p = b.plan(ids, descriptors, torch.zeros(4))
    assert p.metadata['ids'].tolist() == [1,3]
    assert b.history.clock is None
    payload = ids.float()[None, None, :, None]
    pair = p.merge(0, payload, payload, -2)
    b.commit(p, {0: pair}, 1)
    assert b.history.clock == 0.
