"""Transactional bounded KV selection. Only live slots and an eviction watermark persist."""
from dataclasses import dataclass
import torch
import torch.nn.functional as F

@dataclass
class KVPlan:
    owner: object
    generation: int
    metadata: dict
    source: torch.Tensor
    fresh: torch.Tensor
    watermark: int
    layers: dict

    def merge(self, layer, k, v, seq_dim):
        # No mutation here: safe under activation checkpoint recomputation.
        old = self.owner.layers.get(layer)
        nold = 0 if old is None else old[0].shape[seq_dim]
        if old is not None:
            k = torch.cat((old[0].to(k), k), dim=seq_dim)
            v = torch.cat((old[1].to(v), v), dim=seq_dim)
        indexes = torch.where(self.fresh, self.source + nold, self.source).to(k.device)
        return k.index_select(seq_dim, indexes), v.index_select(seq_dim, indexes)

class PersistentKVBank:
    def __init__(self, capacity):
        if int(capacity) < 1:
            raise ValueError("KV capacity must be positive")
        self.capacity = int(capacity)
        self.metadata = None
        self.layers = {}
        self.watermark = -1
        self.generation = 0

    @torch.no_grad()
    def plan(self, ids, descriptors, times, motion=None, contact=None):
        ids = torch.as_tensor(ids, device=descriptors.device, dtype=torch.long).flatten()
        if len(ids) != len(descriptors) or len(ids.unique()) != len(ids):
            raise ValueError("fresh KV IDs must be unique and aligned with descriptors")
        d = F.normalize(descriptors.detach().float(), dim=-1)
        t = torch.as_tensor(times, device=d.device, dtype=torch.float32).flatten()
        m = torch.ones_like(t) if motion is None else torch.as_tensor(motion, device=d.device).flatten().float()
        c = torch.zeros_like(t) if contact is None else torch.as_tensor(contact, device=d.device).flatten().float()
        if any(len(x) != len(ids) for x in (t,m,c)) or not all(torch.isfinite(x).all() for x in (d,t,m,c)) or (c < 0).any():
            raise ValueError("invalid KV scoring metadata")
        old = self.metadata
        oldids = ids[:0] if old is None else old['ids'].to(ids.device)
        # Refreshed survivors replace old versions; evicted old IDs cannot resurrect.
        accept = (ids > self.watermark) | torch.isin(ids, oldids)
        if not len(oldids):
            old = None
        refresh = torch.isin(oldids, ids)
        keepold = (~refresh).nonzero().flatten()
        usefresh = accept.nonzero().flatten()
        newest = ids > self.watermark
        newmeta = dict(ids=ids, d=d, t=t, latest=t.clone(), m=m, c=c)
        if old is not None and len(oldids):
            # Preserve recurrence history for refreshed IDs, without storing tombstones.
            match = ids[:,None] == oldids[None,:]
            found = match.any(-1)
            oi = match.long().argmax(-1)
            newmeta['latest'][found] = old['latest'].to(t.device)[oi[found]]
            newmeta['m'][found] = old['m'].to(t.device)[oi[found]]
            if contact is None:
                newmeta['c'][found] = old['c'].to(t.device)[oi[found]]
        meta = {key: (value[usefresh] if old is None else torch.cat((old[key].to(value.device)[keepold],value[usefresh]))) for key,value in newmeta.items()}
        src = torch.cat((keepold,usefresh))
        fresh = torch.cat((torch.zeros_like(keepold,dtype=torch.bool),torch.ones_like(usefresh,dtype=torch.bool)))
        if len(meta['ids']):
            if newest.any():
                matches = (meta['d'] @ d[newest].T >= .9) & meta['d'].ne(0).any(-1)[:,None] & d[newest].ne(0).any(-1)[None,:]
                seen = torch.where(matches,t[newest][None,:],torch.full_like(matches,float('-inf'),dtype=t.dtype)).amax(-1)
                meta['latest'] = torch.maximum(meta['latest'],seen)
            now = max(float(t.max()) if len(t) else 0., float(meta['t'].max()))
            # Without novelty, N new tokens each score 2 and always evict all N old
            # tokens (C=0). Penalize semantic repetition against live memory only.
            repeats = torch.zeros_like(meta['t'])
            if old is not None and len(oldids):
                similar = (meta['d'] @ old['d'].to(d.device).T >= .9)
                similar &= meta['ids'][:,None] != oldids[None,:]
                repeats = similar.sum(-1).to(t.dtype)
            novelty = 1 / (1 + repeats)
            score = torch.exp(-(now-meta['t']).clamp_min(0)/8) + 1-torch.exp(-meta['c']/8) + novelty*meta['d'].ne(0).any(-1)*torch.exp(-(meta['latest']-meta['t']).clamp_min(0)/8)
            chosen = torch.argsort(score,descending=True,stable=True)[:self.capacity]
            chosen = chosen[torch.argsort(meta['ids'][chosen])]
            meta = {key:value[chosen].clone() for key,value in meta.items()}
            src,fresh = src[chosen],fresh[chosen]
        watermark = max(self.watermark,int(ids.max()) if len(ids) else -1)
        return KVPlan(self,self.generation,meta,src,fresh,watermark,{})

    def commit(self, plan, layers, expected_layers):
        if plan.owner is not self or plan.generation != self.generation:
            raise RuntimeError("stale or foreign KV transaction")
        if set(layers) != set(range(expected_layers)):
            raise RuntimeError("incomplete KV transaction")
        # Allocate everything before publishing; failures leave the bank unchanged.
        saved = {i:(k.detach().clone(),v.detach().clone()) for i,(k,v) in layers.items()}
        self.layers,self.metadata,self.watermark = saved,plan.metadata,plan.watermark
        self.generation += 1


class OpenWAMKVSession:
    """One successful generate call = one bank update, never one denoising step.

    All observed keys use the first-frame RoPE anchor (temporal coordinate zero).
    Their spatial coordinates are preserved. Age is explicit selector metadata.
    The clean-frame branch always self-attends densely to its own N tokens;
    action/future queries attend only to the selected N memory positions.
    """
    def __init__(self, state, descriptors, image, motion_only=False, threshold=.04):
        self.state,self.descriptors,self.image = state,descriptors,image
        self.motion_only,self.threshold = motion_only,threshold
        self.plan = None
        self.layers = {}
        self.n = None

    def prepare(self, h, w, device):
        import numpy as np
        n = int(h*w)
        if self.plan is not None:
            if n != self.n:
                raise ValueError("video token grid changed during denoising")
            return
        bank = self.state.get('bank')
        if bank is None:
            bank = PersistentKVBank(n)
        if bank.capacity != n:
            raise ValueError("frame token budget changed; reset the episode")
        d = self.descriptors.to(device=device,dtype=torch.float32)
        if d.ndim != 4 or d.shape[0] != 1:
            raise ValueError("persistent OpenWAM expects one current DINO grid [1,H,W,D]")
        d = F.adaptive_avg_pool2d(d.permute(0,3,1,2),(h,w)).flatten(2).transpose(1,2)[0]
        rgb = torch.as_tensor(np.asarray(self.image).copy(),device=device).permute(2,0,1).float()/255
        previous = self.state.get('rgb')
        motion = torch.ones(n,device=device) if previous is None else F.adaptive_avg_pool2d((rgb-previous.to(device)).abs().mean(0)[None,None],(h,w)).flatten()
        self.rgb = rgb
        ids = bank.generation*n+torch.arange(n,device=device)
        times = torch.full((n,),float(bank.generation),device=device)
        self.plan = bank.plan(ids,d,times,motion)
        self.bank,self.n = bank,n

    def attention(self, driver, layer, q, k, v, mask):
        n = self.n
        if q.shape[0] != 1 or k.shape[1] != q.shape[1] or mask is None or mask.dtype != torch.bool:
            raise ValueError("persistent OpenWAM requires single-sample joint attention with explicit boolean mask")
        if mask[..., :n,n:].any() or not mask[..., :n,:n].all():
            raise ValueError("clean-frame KV requires first_frame_causal video mask and no clean-frame action attention")
        mk,mv = self.plan.merge(layer,k[:,:n],v[:,:n],1)
        self.layers[layer] = (mk,mv)
        # This branch makes the clean K/V independent of action/future noise.
        clean = driver._mixed_attention(q[:,:n],k[:,:n],v[:,:n],mask[..., :n,:n])
        read = torch.ones(len(self.plan.source),device=k.device,dtype=torch.bool)
        if self.motion_only:
            read = self.plan.metadata['m'].to(k.device) > self.threshold
        indexes = read.nonzero().flatten()
        mk,mv = mk.index_select(1,indexes),mv.index_select(1,indexes)
        # Cross-modality policy is preserved; each selected memory slot has
        # the same visibility as an observed clean-frame token.
        prefix_mask = mask[..., n:,:1].expand(*mask[...,n:,:1].shape[:-1],len(indexes))
        rest_mask = torch.cat((prefix_mask,mask[...,n:,n:]),dim=-1)
        rest = driver._mixed_attention(q[:,n:],torch.cat((mk,k[:,n:]),1),torch.cat((mv,v[:,n:]),1),rest_mask)
        return torch.cat((clean,rest),1)

    def commit(self, expected_layers):
        if self.plan is None:
            raise RuntimeError("persistent KV session was never executed")
        self.bank.commit(self.plan,self.layers,expected_layers)
        self.state['bank'] = self.bank
        self.state['rgb'] = self.rgb.detach().clone()
