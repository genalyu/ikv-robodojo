import importlib.util,pathlib,torch,pytest
p=pathlib.Path(__file__).parents[1]/'opendm/model/dm05/qwen_archive.py'
spec=importlib.util.spec_from_file_location('qwen_archive',p);mod=importlib.util.module_from_spec(spec)
import sys;sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
class Scorer:
 def __init__(self):self.calls=[];self.fail=False
 def score(self,candidates,current_time):
  if self.fail:raise ValueError('invalid Qwen response')
  self.calls.append(candidates);return {c['id']:100-int(c['id']) for c in candidates}
def update(b,ids):
 p=b.plan(torch.tensor(ids),torch.zeros(len(ids),1),torch.full((len(ids),),float(b.generation)))
 k=torch.tensor(ids,dtype=torch.float32).reshape(1,1,-1,1)
 out=p.merge(0,k,k,2);b.commit(p,{0:out},1);return out

def test_archived_old_positions_reenter_and_score_all():
 c=Scorer();b=mod.QwenArchiveBank(2,c,'task',grid=(1,2),protect_current=False)
 update(b,[0,1]);update(b,[2,3]);assert len(b.archive_metadata['ids'])==4
 # Make the previously unselected historical row win.
 c.score=lambda xs,t:{x['id']:100 if x['id']=='2' else 0 for x in xs}
 out=update(b,[4,5]);assert 2 in out[0].flatten().tolist();assert len(b.archive_metadata['ids'])==6
 assert all(k.device.type=='cpu' for pair in b.archive_layers.values() for k in pair)

def test_failure_keeps_archive_and_generation():
 c=Scorer();b=mod.QwenArchiveBank(2,c,'task',grid=(1,2),protect_current=False)
 update(b,[0,1]);c.fail=True
 with pytest.raises(ValueError):update(b,[2,3])
 assert b.generation==1 and b.archive_metadata['ids'].tolist()==[0,1]

def test_protection_and_refresh_do_not_duplicate_archive():
 c=Scorer();b=mod.QwenArchiveBank(4,c,'task',grid=(1,2))
 update(b,[0,1]);update(b,[0,1,2,3]);update(b,[2,3,4,5])
 assert len(b.archive_metadata['ids'])==6
 assert {4,5}<=set(b.metadata['ids'].tolist())

def test_dm05_raw_keys_and_values_preserve_different_precision():
 c=Scorer();b=mod.QwenArchiveBank(2,c,'task',grid=(1,2),protect_current=False)
 for gen in range(2):
  ids=torch.arange(2)+gen*2
  plan=b.plan(ids,torch.zeros(2,1),torch.full((2,),float(gen)))
  k=torch.ones(1,1,2,4,dtype=torch.float32)
  v=torch.ones(1,1,2,4,dtype=torch.bfloat16)
  out=plan.merge(0,k,v,2);b.commit(plan,{0:out},1)
  assert out[0].dtype==torch.float32 and out[1].dtype==torch.bfloat16

def test_all_fit_skips_only_irrelevant_ranking():
 c=Scorer();b=mod.QwenArchiveBank(4,c,'task',grid=(1,2),protect_current=False)
 update(b,[0,1]);update(b,[2,3]);assert c.calls==[]
 assert b.metadata['ids'].tolist()==[0,1,2,3]
 update(b,[4,5]);assert len(c.calls)==1 and len(c.calls[0])==6
