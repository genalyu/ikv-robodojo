"""Task-conditioned Qwen selection with transactional CPU archive of real KV."""
from dataclasses import dataclass
import json,time
import torch

@dataclass
class ArchivePlan:
    owner: object
    generation: int
    metadata: dict
    source: torch.Tensor
    fresh: torch.Tensor
    all_metadata: dict
    keepold: torch.Tensor
    usefresh: torch.Tensor
    staged: dict
    watermark: int = -1

    def merge(self, layer, k, v, seq_dim):
        bank=self.owner
        old=bank.archive_layers.get(layer)
        if old is not None:
            oldk,oldv=old
            kept=(oldk.index_select(seq_dim,self.keepold),oldv.index_select(seq_dim,self.keepold))
            self.staged[layer]=(torch.cat((kept[0],k.detach().cpu().index_select(seq_dim,self.usefresh)),seq_dim),torch.cat((kept[1],v.detach().cpu().index_select(seq_dim,self.usefresh)),seq_dim))
        else:
            self.staged[layer]=(k.detach().cpu().index_select(seq_dim,self.usefresh),v.detach().cpu().index_select(seq_dim,self.usefresh))
        # Only requested historical rows transfer to GPU; never the full archive.
        outk=torch.empty_like(k.narrow(seq_dim,0,0)).new_empty([len(self.source) if j==seq_dim%k.ndim else n for j,n in enumerate(k.shape)])
        outv=v.new_empty([len(self.source) if j==seq_dim%v.ndim else n for j,n in enumerate(v.shape)])
        for isfresh in (False,True):
            destinations=(self.fresh==isfresh).nonzero().flatten()
            if not len(destinations):continue
            src=self.source[destinations]
            if isfresh:
                selectedk=k.index_select(seq_dim,src.to(k.device));selectedv=v.index_select(seq_dim,src.to(v.device))
            else:
                cached=bank.layers.get(layer)
                ids=bank.archive_metadata['ids'][src.cpu()]
                resident_ids=bank.metadata['ids'] if bank.metadata is not None else ids[:0]
                positions=torch.searchsorted(resident_ids,ids).clamp_max(max(len(resident_ids)-1,0))
                resident=(resident_ids[positions]==ids) if len(resident_ids) else torch.zeros(len(ids),dtype=torch.bool)
                if cached is None:resident.zero_()
                for available in (False,True):
                    local=(resident==available).nonzero().flatten()
                    if not len(local):continue
                    dst=destinations[local].to(k.device)
                    pair=cached if available else old
                    take=positions[local] if available else src[local].cpu()
                    outk.index_copy_(seq_dim,dst,pair[0].index_select(seq_dim,take.to(pair[0].device)).to(k))
                    outv.index_copy_(seq_dim,dst,pair[1].index_select(seq_dim,take.to(pair[1].device)).to(v))
                continue
            outk.index_copy_(seq_dim,destinations.to(k.device),selectedk)
            outv.index_copy_(seq_dim,destinations.to(v.device),selectedv)
        return outk,outv

class QwenArchiveBank:
    def __init__(self,capacity,client,task,max_bytes=8*1024**3,audit=None,grid=(4,4),protect_current=True):
        self.capacity=int(capacity);self.client=client;self.task=task
        self.max_bytes=int(max_bytes);self.audit=audit;self.grid=tuple(grid);self.patches=self.grid[0]*self.grid[1];self.protect_current=protect_current
        self.metadata=None;self.archive_metadata=None;self.archive_layers={};self.layers={};self.generation=0
    @torch.no_grad()
    def plan(self,ids,descriptors,times,motion=None,contact=None):
        ids=torch.as_tensor(ids).detach().cpu().long().flatten()
        frames=torch.as_tensor(descriptors).detach().cpu().long().reshape(-1)
        t=torch.as_tensor(times).detach().cpu().float().flatten()
        m=torch.ones_like(t) if motion is None else torch.as_tensor(motion).detach().cpu().float().flatten()
        if len(ids)!=len(frames) or len(t)!=len(ids) or len(ids.unique())!=len(ids):raise ValueError('unaligned Qwen KV index')
        old=self.archive_metadata
        oldids=ids[:0] if old is None else old['ids']
        keepold=(~torch.isin(oldids,ids)).nonzero().flatten();usefresh=torch.arange(len(ids))
        freshmeta=dict(ids=ids,frame=frames,t=t,m=m)
        meta={name:value if old is None else torch.cat((old[name][keepold],value)) for name,value in freshmeta.items()}
        source=torch.cat((keepold,usefresh));fresh=torch.cat((torch.zeros(len(keepold),dtype=torch.bool),torch.ones(len(ids),dtype=torch.bool)))
        candidates=[dict(id=str(int(uid)),time=float(tm),modality='visual',frame=int(frame),patch=int(uid)%self.patches,grid=list(self.grid)) for uid,tm,frame in zip(meta['ids'],meta['t'],meta['frame'])]
        before=time.perf_counter()
        contended=len(candidates)>self.capacity
        scores=self.client.score(candidates,float(t.max()) if len(t) else 0.) if contended else {}
        if len(candidates)<=self.capacity:
            chosen=torch.argsort(meta['ids'])
        else:
            score=torch.tensor([scores[str(int(i))] for i in meta['ids']],dtype=torch.float32)
            # Protect the newest real history frame; current camera KV is dense outside this bank.
            protected=(meta['t']==meta['t'].max()) if self.protect_current else torch.zeros_like(meta['t'],dtype=torch.bool)
            if int(protected.sum())>self.capacity:raise ValueError('current observation exceeds GPU budget')
            order=torch.argsort(score,descending=True,stable=True)
            chosen=torch.cat((protected.nonzero().flatten(),order[~protected[order]][:self.capacity-int(protected.sum())]))
            chosen=chosen[torch.argsort(meta['ids'][chosen])]
        if self.audit:
            with open(self.audit,'a') as f:f.write(json.dumps(dict(generation=self.generation,candidates=len(candidates),selected=len(chosen),scoring_seconds=time.perf_counter()-before,scored=contended,ids=meta['ids'][chosen].tolist()))+'\n')
        return ArchivePlan(self,self.generation,{k:v[chosen] for k,v in meta.items()},source[chosen],fresh[chosen],meta,keepold,usefresh,{},-1)
    def commit(self,plan,layers,expected_layers):
        if plan.owner is not self or plan.generation!=self.generation:raise RuntimeError('stale archive transaction')
        if set(plan.staged)!=set(range(expected_layers)):raise RuntimeError('incomplete archive layers')
        size=sum(x.numel()*x.element_size() for pair in plan.staged.values() for x in pair)
        if size>self.max_bytes:raise MemoryError('CPU KV archive capacity exceeded')
        self.archive_layers=plan.staged;self.archive_metadata=plan.all_metadata
        self.layers={i:(k.detach(),v.detach()) for i,(k,v) in layers.items()}
        self.metadata=plan.metadata;self.generation+=1
