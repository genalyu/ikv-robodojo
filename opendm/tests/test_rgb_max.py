import torch
import pytest
from opendm.model.dm05.rgb_max import PeriodicRGBBank,RGBMaxClient
class Scorer:
    def __init__(self):self.calls=0
    def score(self,rows,now):
        self.calls+=1
        return {r['id']:100 if int(r['id'])<16 else 1 for r in rows}
def test_periodic_fifo_archive_recall_and_transaction():
    client=Scorer();bank=PeriodicRGBBank(32,client,'task',interval=4)
    for t in range(9):
        ids=torch.arange(max(0,t-2)*16,(t+1)*16)
        plan=bank.plan(ids,(ids//16)[:,None],(ids//16).float())
        if t==3:assert plan.metadata['ids'].min()==32 and client.calls==0
        if t==4:
            assert plan.metadata['ids'].min()==0 and client.calls==1
            assert not plan.fresh.all()
            assert bank.last_management==0
        pair=torch.arange(len(ids),dtype=torch.float32).reshape(1,1,-1,1)
        merged=plan.merge(0,pair,pair,-2)
        bank.commit(plan,{0:merged},1)
        assert len(bank.metadata['ids'])<=32
        assert len(bank.archive_metadata['ids'])==(t+1)*16
    assert client.calls==2
    stale=bank.plan(ids,(ids//16)[:,None],(ids//16).float())
    with pytest.raises(RuntimeError,match='incomplete'):bank.commit(stale,{},1)
    assert bank.last_management==8

def test_full_assignment_and_invalid_bounds():
    obj={'default_score':1,'blocks':[{'frames':[0,1],'camera':0,'rows':[2,4],'cols':[4,6],'score':99}]}
    a=RGBMaxClient.decode(obj,2)
    assert a.shape==(2,3,8,8) and a[0,0,2,4]==99 and a[1].max()==1
    obj['blocks'][0]['frames']=[0,3]
    with pytest.raises(ValueError):RGBMaxClient.decode(obj,2)
    obj['blocks'][0]['frames']=[0,1];obj['default_score']=True
    with pytest.raises(ValueError):RGBMaxClient.decode(obj,2)
