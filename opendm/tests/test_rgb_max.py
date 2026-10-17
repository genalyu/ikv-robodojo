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

def test_invalid_teacher_assignment_gets_feedback_before_use(tmp_path):
    import json
    import numpy as np
    from unittest.mock import patch
    (tmp_path/'private').mkdir();(tmp_path/'private/bailian.key').write_text('test-only')
    (tmp_path/'prompt_visual.txt').write_text('Score the supplied evidence.')
    client=RGBMaxClient('task',tmp_path)
    client.register(0,[np.zeros((32,32,3),dtype=np.uint8)]*3)
    invalid={'default_score':1,'blocks':[{'frames':[0,0],'camera':0,'rows':[0,1],'cols':[0,1],'score':90}]}
    valid={'default_score':1,'blocks':[{'frames':[0,1],'camera':0,'rows':[0,2],'cols':[0,2],'score':90}]}
    requests=[]
    class Response:
        status_code=200
        def __init__(self,assignment):self.assignment=assignment
        def json(self):return {'model':'test','id':'test','choices':[{'finish_reason':'stop','message':{'content':json.dumps(self.assignment)}}]}
    def transport(*args,**kwargs):
        requests.append(json.loads(json.dumps(kwargs['json'])))
        return Response(invalid if len(requests)==1 else valid)
    candidates=[dict(id=str(i),time=0.,modality='visual',frame=0,patch=i,grid=[4,4]) for i in range(16)]
    with patch('requests.post',side_effect=transport),patch('time.sleep'):
        scores=client.score(candidates,0.)
    assert len(requests)==2
    assert len(requests[0]['messages'])==2
    assert 'empty bounds' in requests[1]['messages'][2]['content']
    assert scores['0']==90 and scores['15']==1
    assert len(list((tmp_path/'requests').glob('*/query*/meta.json')))==1
