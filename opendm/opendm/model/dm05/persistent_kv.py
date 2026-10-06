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


def _rotate(k, rotary, positions, layer_type, inverse=False):
    # Evaluate both local/global Gemma RoPE, including any attention scaling.
    cos,sin = rotary(k,positions,layer_type)
    cos,sin = cos.unsqueeze(1).float(),sin.unsqueeze(1).float()
    x = k.float()
    half = torch.cat((-x[...,x.shape[-1]//2:],x[...,:x.shape[-1]//2]),dim=-1)
    result = (x*cos-half*sin)/(cos.square()+sin.square()) if inverse else x*cos+half*sin
    return result if inverse else result.to(k.dtype)


def splice_dm05_kv(model, cache, input_ids, attention_mask, history_mask, request):
    """Refresh window KV, select persistent slots, and rebase RoPE to prefix positions.

    Prefix VLM computation is unchanged. Only the action expert reads the merged
    memory. Evicted old patches may be contextual input in the refreshed window,
    but can never reenter the bank as standalone cached positions.
    """
    from opendm.model.dm05.dm05_utils import mask_history_pad_tokens_in_attention
    if input_ids.shape[0] != 1:
        raise ValueError("streaming DM05 KV requires one independent bank per sample")
    bank = request['bank']
    slots = ((input_ids[0] == 6)|(input_ids[0] == 7)).nonzero().flatten()
    if len(slots) != bank.capacity:
        raise ValueError("DM05 reserved history slots must equal the KV capacity")
    fresh_positions = history_mask[0].nonzero().flatten()
    if len(fresh_positions) != len(request['ids']):
        raise ValueError("history KV IDs do not match encoded history positions")
    plan = bank.plan(request['ids'],request['descriptors'],request['times'],request.get('motion'),request.get('contact'))
    n = len(plan.source)
    ids = input_ids.clone()
    ids[:,slots] = 7
    dest = slots[-n:] if n else slots[:0]
    ids[:,dest] = 6
    if request.get('motion_only',False) and n:
        ids[:,dest[plan.metadata['m'].to(dest.device) <= request.get('motion_threshold',.04)]] = 7
    oldpos = (mask_history_pad_tokens_in_attention(input_ids,attention_mask).cumsum(-1)-1).clamp_min(0)
    newpos = (mask_history_pad_tokens_in_attention(ids,attention_mask).cumsum(-1)-1).clamp_min(0)
    lm = model.model.vlm.model.language_model
    layers = {}
    for i,layer in enumerate(cache.layers):
        if layer.keys.shape[-2] != input_ids.shape[1]:
            raise ValueError("persistent KV requires full prefill cache, including sliding layers")
        kind = lm.config.layer_types[i]
        key_dtype = layer.keys.dtype
        raw = _rotate(layer.keys,lm.rotary_emb,oldpos,kind,inverse=True)
        freshk = raw.index_select(-2,fresh_positions)
        freshv = layer.values.index_select(-2,fresh_positions)
        mk,mv = plan.merge(i,freshk,freshv,-2)
        layers[i] = (mk,mv)
        # index_copy is differentiable for refreshed K/V; old memory is detached.
        raw = raw.index_copy(-2,dest,mk)
        layer.keys = _rotate(raw,lm.rotary_emb,newpos,kind).to(key_dtype)
        layer.values = layer.values.index_copy(-2,dest,mv)
    return ids,(bank,plan,layers,len(cache.layers))


def commit_dm05(transaction):
    if transaction is not None:
        bank,plan,layers,count = transaction
        bank.commit(plan,layers,count)
