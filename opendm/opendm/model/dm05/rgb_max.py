"""Full observed RGB teacher protocol with periodic transactional KV selection."""
import base64,io,json,time,hashlib
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import requests,torch
from .qwen_archive import QwenArchiveBank,ArchivePlan

class RGBMaxClient:
    def __init__(self,task,root,pooling="mean"):
        self.pooling=pooling
        if pooling not in ("mean","max"):raise ValueError("unknown score pooling")
        self.task=task;self.root=Path(root);self.frames={};self.session=hashlib.sha256((task+str(time.time_ns())).encode()).hexdigest()[:16]
        self.out=self.root/'requests'/self.session;self.out.mkdir(parents=True)
        self.key=(self.root/'private/bailian.key').read_text().strip()
        self.prompt=(self.root/'prompt_visual.txt').read_text().split('[Canonical Output Requirements]')[0]+'''\n[Output Requirements]\nReturn ONLY JSON with exactly default_score (integer 0..100) and blocks (array). Each block has EXACT keys frames:[start,end_exclusive],camera:integer 0/1/2,rows:[start,end_exclusive],cols:[start,end_exclusive],score:integer 0..100. Frames are ORIGINAL observed frame ordinals. Rows and columns range 0..8; bounds are half-open. Initialize every candidate to default_score and apply blocks in order; later blocks overwrite. Cover all frames/cameras/8x8 patches. No prose or Markdown. Group only equally useful evidence. Every request reassesses the whole observed prefix.\n'''
    def register(self,frame,images):
        frame=int(frame)
        if frame in self.frames:return
        atlas=Image.new('RGB',(864,304),'#20242c');draw=ImageDraw.Draw(atlas)
        for c,image in enumerate(images):
            image=Image.fromarray(np.asarray(image,dtype=np.uint8)).resize((256,256),Image.Resampling.BOX)
            x=c*288+32;y=48;atlas.paste(image,(x,y));draw.text((x,2),f'frame={frame} camera={c}',fill='white')
            for k in range(8):
                draw.text((x+k*32+8,28),str(k),fill='white');draw.text((x-23,y+k*32+9),str(k),fill='white')
                if k:draw.line((x+k*32,y,x+k*32,y+255),fill='#737373');draw.line((x,y+k*32,x+255,y+k*32),fill='#737373')
        p=self.out/f'frame_{frame:06d}.jpg';atlas.save(p,quality=93,subsampling=0);self.frames[frame]=p
    @staticmethod
    def decode(obj,n):
        def integer(v,lo,hi):
            if type(v)is not int or not lo<=v<=hi:raise ValueError('invalid integer or score')
            return v
        if type(obj)is not dict or set(obj)!={'default_score','blocks'} or type(obj['blocks'])is not list:raise ValueError('invalid assignment')
        a=np.full((n,3,8,8),integer(obj['default_score'],0,100),dtype=np.uint8)
        for b in obj['blocks']:
            if type(b)is not dict or set(b)!={'frames','camera','rows','cols','score'}:raise ValueError('invalid block')
            bounds=[]
            for name,limit in [('frames',n),('rows',8),('cols',8)]:
                v=b[name]
                if type(v)is not list or len(v)!=2:raise ValueError('invalid bounds')
                start,end=integer(v[0],0,limit),integer(v[1],0,limit)
                if start>=end:raise ValueError('empty bounds')
                bounds.append(slice(start,end))
            a[bounds[0],integer(b['camera'],0,2),bounds[1],bounds[2]]=integer(b['score'],0,100)
        return a
    def score(self,candidates,current_time):
        n=max(self.frames)+1
        if set(self.frames)!=set(range(n)):raise ValueError('RGB prefix has missing frames')
        content=[]
        for j in range(n):
            data=base64.b64encode(self.frames[j].read_bytes()).decode()
            content.append({'type':'image_url','image_url':{'url':'data:image/jpeg;base64,'+data}})
        user=dict(task=self.task,current_frame=n-1,current_time=n-1,fps=1,images='Chronological original RGB frames; each atlas has cameras top/wrist_l/wrist_r, full-field 256x256, grid 8x8. All observed frames included; no future frames.',grid=dict(rows=8,columns=8,camera_order=['top','wrist_l','wrist_r']),visual_candidates=dict(domain=f'frame 0..{n-1}, camera 0..2, row/column 0..7',total=n*192),scoring_target='At the current frame reassess EVERY historical patch for the robot next action. Do not merely score the latest frame.')
        content.append({'type':'text','text':json.dumps(user,ensure_ascii=False)})
        body=dict(model='qwen3.8-max',messages=[dict(role='system',content=self.prompt),dict(role='user',content=content)],temperature=0,enable_thinking=True,thinking_budget=512,max_tokens=16384)
        out=self.out/f'query_{n-1:06d}_{time.time_ns()}';out.mkdir()
        (out/'input.json').write_text(json.dumps(dict(user=user,candidates=candidates,prompt_sha256=hashlib.sha256(self.prompt.encode()).hexdigest(),mapping='native square-padded head camera 8x8 -> 4x4 '+self.pooling+' over each 2x2 block',frame_paths=[str(self.frames[j]) for j in range(n)])))
        start=time.monotonic()
        for attempt in range(4):
            try:
                response=requests.post('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',headers={'Authorization':'Bearer '+self.key},json=body,timeout=(30,480))
                if response.status_code!=200:raise ValueError(f'API HTTP {response.status_code}: {response.text[:300]}')
                result=response.json();(out/f'response_{attempt}.json').write_text(json.dumps(result,ensure_ascii=False))
                choice=result['choices'][0]
                if choice['finish_reason']!='stop':raise ValueError('incomplete response')
                text=choice['message']['content'].strip()
                if text.startswith('```json') and text.endswith('```'):text=text[7:-3].strip()
                obj=json.loads(text);scores=self.decode(obj,n)
                if scores.min()==scores.max():raise ValueError('constant scores')
                blocks=scores[:,0].reshape(n,4,2,4,2)
                pooled=getattr(blocks,self.pooling)(axis=(2,4))
                answer={c['id']:float(pooled[int(c['frame']),int(c['patch'])//4,int(c['patch'])%4]) for c in candidates}
                (out/'assignment.json').write_text(json.dumps(obj));np.savez_compressed(out/'scores.npz',scores=scores,dm05_head_scores=pooled)
                (out/'meta.json').write_text(json.dumps(dict(model=result.get('model'),request_id=result.get('id'),usage=result.get('usage'),elapsed_seconds=time.monotonic()-start,frames=n,candidates=len(candidates))))
                return answer
            except Exception as e:
                (out/f'error_{attempt}.json').write_text(json.dumps(dict(error=str(e))))
                if attempt==3:raise
                body['messages']=body['messages'][:2]+[dict(role='user',content='Your previous response failed strict validation: '+str(e)+'. Re-evaluate the entire assignment. Every block must have EXACT keys frames,camera,rows,cols,score. Frames/rows/cols are HALF-OPEN [start,end_exclusive]; end MUST be greater than start. A single frame f is [f,f+1], never [f,f]. No extra fields. Do not omit historical frames. Return complete valid JSON only.')]
                time.sleep(min(20,3*2**attempt))
        raise RuntimeError('unreachable')

class PeriodicRGBBank(QwenArchiveBank):
    def __init__(self,capacity,client,task,interval=4,intermediate_policy="cached_score",**kwargs):
        super().__init__(capacity,client,task,protect_current=False,**kwargs)
        if type(interval)is not int or interval<2:raise ValueError('enabled interval must be >=2')
        if intermediate_policy not in ("fifo","cached_score"):raise ValueError("unknown intermediate eviction policy")
        self.interval=interval;self.last_management=None
        self.intermediate_policy=intermediate_policy;self.cached_scores={}
    @torch.no_grad()
    def plan(self,ids,descriptors,times,motion=None,contact=None):
        ids=torch.as_tensor(ids).detach().cpu().long().flatten();frames=torch.as_tensor(descriptors).detach().cpu().long().flatten();t=torch.as_tensor(times).detach().cpu().float().flatten()
        if len(ids)!=len(frames) or len(ids)!=len(t) or len(ids.unique())!=len(ids):raise ValueError('unaligned RGB KV index')
        old=self.archive_metadata;oldids=ids[:0] if old is None else old['ids'];keepold=(~torch.isin(oldids,ids)).nonzero().flatten();usefresh=torch.arange(len(ids))
        freshmeta=dict(ids=ids,frame=frames,t=t,m=torch.ones_like(t))
        meta={k:v if old is None else torch.cat((old[k][keepold],v)) for k,v in freshmeta.items()}
        source=torch.cat((keepold,usefresh));fresh=torch.cat((torch.zeros(len(keepold),dtype=torch.bool),torch.ones(len(ids),dtype=torch.bool)))
        now=float(t.max()) if len(t) else 0.;due=self.last_management is None or now-self.last_management>=self.interval
        contended=len(meta['ids'])>self.capacity;before=time.monotonic()
        next_scores=dict(self.cached_scores);protected_new=0
        # Unscored evidence cannot all fit: obtain real scores rather than
        # inventing low scores or silently discarding protected observations.
        if contended and self.intermediate_policy=="cached_score" and not next_scores:due=True
        if due and contended:
            candidates=[dict(id=str(int(uid)),time=float(tm),modality='visual',frame=int(frame),patch=int(uid)%16,grid=[4,4]) for uid,tm,frame in zip(meta['ids'],meta['t'],meta['frame'])]
            scores=self.client.score(candidates,now)
            next_scores={int(uid):float(scores[str(int(uid))]) for uid in meta['ids']}
            if any(not np.isfinite(v) or not 0<=v<=100 for v in next_scores.values()):raise ValueError('invalid cached importance score')
            values=torch.tensor([next_scores[int(uid)] for uid in meta['ids']])
            recent=torch.argsort(meta['t'],descending=True,stable=True);chosen=recent[torch.argsort(values[recent],descending=True,stable=True)[:self.capacity]]
        elif due or not contended:chosen=torch.argsort(meta['ids'])[-self.capacity:]
        else:
            # Refresh resident tokens and admit genuinely new IDs; never recall
            # evicted scored IDs between full-archive management events.
            resident=ids[:0] if self.metadata is None else self.metadata['ids']
            newest=-1 if old is None or not len(oldids) else int(oldids.max())
            eligible=torch.isin(meta['ids'],resident)|(meta['ids']>newest)
            chosen=eligible.nonzero().flatten()
            if self.intermediate_policy=="fifo":
                chosen=chosen[torch.argsort(meta['ids'][chosen])][-self.capacity:]
            else:
                unscored=torch.tensor([int(meta['ids'][i]) not in next_scores for i in chosen],dtype=torch.bool)
                protected=chosen[unscored];protected_new=len(protected)
                if protected_new>self.capacity:raise ValueError('unscored observations exceed capacity; shorten scoring interval')
                ranked=chosen[~unscored]
                ranked=ranked[torch.argsort(meta['t'][ranked],descending=True,stable=True)]
                values=torch.tensor([next_scores[int(meta['ids'][i])] for i in ranked])
                ranked=ranked[torch.argsort(values,descending=True,stable=True)]
                chosen=torch.cat((protected,ranked[:self.capacity-protected_new]))
        chosen=chosen[torch.argsort(meta['ids'][chosen])]
        plan=ArchivePlan(self,self.generation,{k:v[chosen] for k,v in meta.items()},source[chosen],fresh[chosen],meta,keepold,usefresh,{},-1);plan.management_time=now if due else self.last_management;plan.cached_scores=next_scores
        if self.audit:
            with open(self.audit,'a') as f:f.write(json.dumps(dict(generation=self.generation,now=now,interval=self.interval,management_due=due,intermediate_policy=self.intermediate_policy,protected_unscored=protected_new,scored=bool(due and contended),candidates=len(meta['ids']),selected=len(chosen),ids=meta['ids'][chosen].tolist(),scoring_seconds=time.monotonic()-before))+'\n')
        return plan
    def commit(self,plan,layers,expected_layers):
        super().commit(plan,layers,expected_layers);self.last_management=plan.management_time;self.cached_scores=plan.cached_scores
