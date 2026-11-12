"""Pure T+C+R+Q scoring with CPU historical KV archive, no generative scorer."""
import json,time
from types import SimpleNamespace
from functools import lru_cache
import torch
import torch.nn.functional as F
from .qwen_archive import QwenArchiveBank,ArchivePlan,resolve_query_time
from .formula_content import ContentHistory

@lru_cache(maxsize=2)
def semantic_model(path,device):
    from transformers import AutoModel,AutoTokenizer,AutoImageProcessor
    model=AutoModel.from_pretrained(path,local_files_only=True).to(device).eval().requires_grad_(False)
    return model,AutoTokenizer.from_pretrained(path,local_files_only=True),AutoImageProcessor.from_pretrained(path,local_files_only=True)

class FormulaIndex:
    def __init__(self,path,task,device,bias=0.,temperature=.07):
        self.model,self.tokenizer,self.processor=semantic_model(path,str(device))
        self.device=device;self.bias=bias;self.temperature=temperature
        x=self.tokenizer([task],padding="max_length",truncation=True,max_length=self.model.config.text_config.max_position_embeddings,return_tensors="pt").to(device)
        with torch.inference_mode():
            text=self.model.get_text_features(**x)
            if not isinstance(text,torch.Tensor):text=text.pooler_output
            self.text=F.normalize(text.float(),dim=-1)
    @torch.inference_mode()
    def relevance(self,image):
        import numpy as np
        rgb=torch.from_numpy(np.asarray(image).copy()).permute(2,0,1)[None].to(self.device).float()/255
        size=self.model.config.vision_config.image_size
        rgb=F.interpolate(rgb,(size,size),mode="bicubic",align_corners=False,antialias=True).clamp(0,1).contiguous()
        mean=rgb.new_tensor(self.processor.image_mean)[None,:,None,None];std=rgb.new_tensor(self.processor.image_std)[None,:,None,None]
        raw=self.model.vision_model(pixel_values=(rgb-mean)/std,return_dict=True).last_hidden_state
        aligned=self.model.vision_model.head(raw.flatten(0,1)[:,None]).reshape_as(raw)
        cosine=(F.normalize(aligned.float(),dim=-1)*self.text[:,None]).sum(-1)
        side=size//self.model.config.vision_config.patch_size
        q=torch.sigmoid((cosine-self.bias)/self.temperature).reshape(1,1,side,side)
        return F.adaptive_avg_pool2d(q,(4,4)).flatten().cpu()

class FormulaArchiveBank(QwenArchiveBank):
    def __init__(self,capacity,audit=None,weights=(1.,1.,1.,1.),scales=(8.,8.,8.),threshold=.9):
        super().__init__(capacity,None,"",audit=audit,protect_current=False)
        self.weights=weights;self.scales=scales
        self.history=ContentHistory(SimpleNamespace(content_threshold=threshold,content_capacity=2048,persistence_scale=8.,class_recency_scale=scales[2]),"cpu")
    @torch.no_grad()
    def plan(self,ids,descriptors,times,motion=None,contact=None,query_time=None):
        ids=torch.as_tensor(ids).detach().cpu().long().flatten()
        features=descriptors.detach().float().cpu()
        t=torch.as_tensor(times).detach().cpu().float().flatten()
        d=F.normalize(features[:,:-1],dim=-1);q=features[:,-1]
        c=torch.zeros_like(t) if contact is None else torch.as_tensor(contact).cpu().float().flatten()
        if len(ids)!=len(d) or len(ids)!=len(t) or len(ids.unique())!=len(ids) or not torch.isfinite(features).all():raise ValueError("invalid formula indices")
        old=self.archive_metadata;oldids=ids[:0] if old is None else old["ids"]
        # An initial empty step has no descriptor-width contract yet.
        if not len(oldids): old=None
        keepold=(~torch.isin(oldids,ids)).nonzero().flatten();usefresh=torch.arange(len(ids))
        freshmeta=dict(ids=ids,d=d,q=q,c=c,t=t)
        meta={k:v if old is None else torch.cat((old[k][keepold],v)) for k,v in freshmeta.items()}
        source=torch.cat((keepold,usefresh));fresh=torch.cat((torch.zeros(len(keepold),dtype=torch.bool),torch.ones(len(ids),dtype=torch.bool)))
        before=time.perf_counter();snapshot=self.history.snapshot()
        try:
            self.history.observe(d,t,torch.ones_like(t))
            now=resolve_query_time(t,meta["t"],query_time)
            T=torch.exp(-(now-meta["t"]).clamp_min(0)/self.scales[0])
            C=1-torch.exp(-meta["c"]/self.scales[1])
            R=self.history.class_recency(meta["d"],meta["t"])
            Q=meta["q"]
            score=sum(w*x for w,x in zip(self.weights,(T,C,R,Q)))
            recent=torch.argsort(meta["t"],descending=True,stable=True)
            chosen=recent[torch.argsort(score[recent],descending=True,stable=True)][:self.capacity]
            chosen=chosen[torch.argsort(meta["ids"][chosen])]
            plan=ArchivePlan(self,self.generation,{k:v[chosen] for k,v in meta.items()},source[chosen],fresh[chosen],meta,keepold,usefresh,{},-1)
            plan.content_state=self.history.snapshot()
        finally:self.history.restore(snapshot)
        if self.audit:
            row=dict(generation=self.generation,candidates=len(meta["ids"]),selected=len(chosen),scoring_seconds=time.perf_counter()-before,ids=meta["ids"][chosen].tolist(),score_min=float(score.min()) if len(score) else None,score_max=float(score.max()) if len(score) else None,weights=self.weights,scales=self.scales,terms={k:dict(min=float(v.min()) if len(v) else None,max=float(v.max()) if len(v) else None) for k,v in zip(("T","C","R","Q"),(T,C,R,Q))})
            with open(self.audit,"a") as f:f.write(json.dumps(row)+"\n")
        return plan
    def commit(self,plan,layers,expected_layers):
        super().commit(plan,layers,expected_layers)
        self.history.restore(plan.content_state)
