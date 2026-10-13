"""Patch importance from immutable native local and full-scene Qwen features."""
import base64,hashlib,io,json,time,math,copy
from pathlib import Path
import torch
from PIL import Image
from transformers.modeling_outputs import BaseModelOutputWithPooling
from .qwen_sys2 import QwenSys2 as Base
from .sys2 import validate_candidates,validate_scores
from .score_json import _DATA
from lmformatenforcer import JsonSchemaParser
from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data,build_transformers_prefix_allowed_tokens_fn

def branch_cache(cache):
    # Dynamic attention layers append with torch.cat; old keys/values stay immutable.
    # Linear-attention layers mutate convolution/recurrent buffers: clone those.
    result=copy.copy(cache);result.layers=[]
    def state(value):
        if isinstance(value,torch.Tensor):return value.clone()
        if isinstance(value,list):return [state(x) for x in value]
        if isinstance(value,dict):return {k:state(v) for k,v in value.items()}
        return copy.copy(value)
    for layer in cache.layers:
        branch=copy.copy(layer)
        for name,value in vars(layer).items():
            if name in ("keys","values"):continue
            if isinstance(value,(torch.Tensor,list,dict)):setattr(branch,name,state(value))
        result.layers.append(branch)
    return result

def observation_times(frames,context,candidates):
    """Deduplicate feature storage without dropping repeated observation events."""
    events={f:set() for f in frames}
    for row in context:
        if row["frame"] not in events:raise ValueError("context references an unknown frame")
        value=row["time"]
        if type(value) not in (int,float) or not math.isfinite(value):raise ValueError("invalid observation time")
        events[row["frame"]].add(value)
    if any(not values for values in events.values()):raise ValueError("context omits a frame timeline")
    for candidate in candidates:
        if candidate["time"] not in events[candidate["frame"]]:
            raise ValueError("context omits candidate observation time")
    return {f:sorted(values) for f,values in events.items()}

class QwenSys2(Base):
    protocol="native_roi_reasoned_v1"
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.patch_prompt=Path(__file__).with_name("roi_reasoned_prompt.txt").read_text()

        self.scorer_revision=hashlib.sha256(Path(__file__).read_bytes()+self.patch_prompt.encode()+self.provenance.encode()).hexdigest()
        self.feature_bytes=sum(p.stat().st_size for p in self.directory.rglob("*.pt"))
    def roi_path(self,frame,grid):
        h,w=grid
        return self.directory/"rois"/str(frame)/f"{h}x{w}.pt"
    @torch.inference_mode()
    def encode(self,encoded_image,grid=(4,4)):
        grid=tuple(grid)
        if len(grid)!=2 or any(type(x) is not int or not 0<x<=32 for x in grid):raise ValueError("invalid native policy grid")
        result=super().encode(encoded_image);frame=result["frame"];path=self.roi_path(frame,grid)
        if path.exists():
            packet=torch.load(path,map_location="cpu",weights_only=True)
            if packet["provenance"]!=self.provenance or tuple(packet["policy_grid"])!=grid:raise ValueError("ROI provenance mismatch")
            return result
        image=Image.open(io.BytesIO(base64.b64decode(encoded_image,validate=True))).convert("RGB")
        width,height=image.size;h,w=grid
        if width<w or height<h:raise ValueError("policy grid finer than source image")
        crops=[image.crop((round(c*width/w),round(r*height/h),round((c+1)*width/w),round((r+1)*height/h))) for r in range(h) for c in range(w)]
        features=[];grids=[]
        for start in range(0,len(crops),8):
            x=self.processor.image_processor(images=crops[start:start+8],return_tensors="pt",size={"shortest_edge":16*32*32,"longest_edge":16*32*32})
            output=self.model.model.get_image_features(x["pixel_values"].to(self.device),x["image_grid_thw"].to(self.device),return_dict=True)
            features.extend(z.detach().cpu() for z in output.pooler_output);grids.append(x["image_grid_thw"].cpu())
        packet=dict(provenance=self.provenance,frame=frame,policy_grid=list(grid),features=features,grids=torch.cat(grids))
        buffer=io.BytesIO();torch.save(packet,buffer);content=buffer.getvalue()
        if self.feature_bytes+len(content)>self.max_feature_bytes:raise MemoryError("Qwen feature index capacity exceeded")
        path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix(".tmp")
        try:tmp.write_bytes(content);tmp.replace(path)
        finally:tmp.unlink(missing_ok=True)
        self.feature_bytes+=len(content)
        return result
    @torch.inference_mode()
    def prefill_prefix(self, ids, features, grid):
        # Full history is retained. Chunk only decoder computation to bound
        # temporary linear-attention activations on a 24 GiB GPU.
        limit=getattr(self,"prefill_chunk_size",2048)
        self.model.model.rope_deltas=None
        positions=self.model._prepare_position_ids_for_generation(ids,dict(
            attention_mask=torch.ones_like(ids),
            mm_token_type_ids=(ids==self.model.config.image_token_id).long(),
            image_grid_thw=grid))
        embeds=self.model.model.get_input_embeddings()(ids)
        visual=torch.cat(features).to(embeds)
        mask,_=self.model.model.get_placeholder_mask(ids,embeds,image_features=visual)
        embeds=embeds.masked_scatter(mask,visual)
        cache=None
        for start in range(0,ids.shape[1],limit):
            end=min(start+limit,ids.shape[1])
            output=self.model.model.language_model(inputs_embeds=embeds[:,start:end],
                position_ids=positions[...,start:end],attention_mask=torch.ones_like(ids[:,:end]),
                past_key_values=cache,use_cache=True)
            cache=output.past_key_values
            del output
        return cache

    @torch.inference_mode()
    def score(self,request):
        started=time.monotonic();candidates=validate_candidates(request["candidates"])
        if not candidates:return dict(visual_scores=[],tactile_scores=[])
        if any(c["modality"]!="visual" for c in candidates):raise ValueError("only observed visual policy rows are supported")
        if not isinstance(request.get("task"),str) or not request["task"].strip():raise ValueError("empty task")
        frames=request.get("context_frames",list(dict.fromkeys(c["frame"] for c in candidates)))
        if any(type(f) is not int or f<0 for f in frames) or len(frames)!=len(set(frames)) or not set(c["frame"] for c in candidates)<=set(frames):raise ValueError("invalid context references")
        context=request.get("context_index",[dict(frame=c["frame"],time=c["time"]) for c in candidates])
        times=observation_times(frames,context,candidates)
        if type(request.get("current_time")) not in (int,float) or not math.isfinite(request["current_time"]):raise ValueError("invalid current time")
        if any(t>request["current_time"] for values in times.values() for t in values):raise ValueError("future evidence in scoring context")
        frames=sorted(frames,key=lambda f:(times[f][0],f));aliases={f:i for i,f in enumerate(frames)}
        content=[];features=[];grids=[];rois={}
        for f in frames:
            packet=torch.load(self.directory/f"{f}.pt",map_location="cpu",weights_only=True)
            if packet["provenance"]!=self.provenance:raise ValueError("full feature provenance mismatch")
            features.append(packet["features"].to(self.device));grids.append(packet["grid"])
            content.extend([dict(type="text",text=f"Full scene {aliases[f]}, observed at times {json.dumps(times[f])}:"),dict(type="text",text="<|vision_start|>"+"<|image_pad|>"*len(features[-1])+"<|vision_end|>")])
        content.append(dict(type="text",text=f"Task: {request['task']}\nCurrent observation time: {request['current_time']}. Assess only evidence in the candidate local image that follows.\n"))
        tokenizer=self.processor.tokenizer
        messages=[dict(role="system",content=self.patch_prompt),dict(role="user",content=content)]
        rendered=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
        ending="<|im_end|>\n";assert rendered.endswith(ending)
        prefix=rendered[:-len(ending)]
        ids0=tokenizer(prefix,return_tensors="pt",add_special_tokens=False)["input_ids"].to(self.device);length=ids0.shape[1]
        if length+1024>self.max_context:raise ValueError("complete history exceeds scoring context")
        all_grid=torch.cat(grids).to(self.device)
        native=dict(image=BaseModelOutputWithPooling(pooler_output=tuple(features)))
        base_cache=self.prefill_prefix(ids0,features,all_grid)
        key=id(tokenizer)
        if key not in _DATA:_DATA[key]=build_token_enforcer_tokenizer_data(tokenizer)
        result=dict(visual_scores=[],tactile_scores=[]);diagnostics=[];memo={}
        for n,c in enumerate(sorted(candidates,key=lambda c:(times[c["frame"]][0],c["frame"],c["time"],tuple(c["grid"]),c["patch"],c["id"]))):
            grid=tuple(c["grid"]);roi_key=(c["frame"],grid)
            if roi_key not in rois:
                rp=torch.load(self.roi_path(*roi_key),map_location="cpu",weights_only=True)
                if rp["provenance"]!=self.provenance or tuple(rp["policy_grid"])!=grid:raise ValueError("local feature provenance mismatch")
                rois[roi_key]=rp
            rp=rois[roi_key];z=rp["features"][c["patch"]].to(self.device);g=rp["grids"][c["patch"]:c["patch"]+1].to(self.device)
            identity=(c["frame"],c["time"],grid,c["patch"])
            if identity not in memo:
                row,col=divmod(c["patch"],grid[1])
                box=[round(col*1000/grid[1]),round(row*1000/grid[0]),round((col+1)*1000/grid[1]),round((row+1)*1000/grid[0])]
                query=(f"Specific task: {request['task']}\nCurrent observation time: {request['current_time']}.\n"
                    f"This is ONLY patch {c['patch']} of a {grid[0]}x{grid[1]} policy grid from scene {aliases[c['frame']]}, observation time {c['time']}. Its row={row}, column={col}, box_norm_1000={box} [left,top,right,bottom] within that full scene. The local crop below is the only evidence to score. Use the stated location to link any current cover to historical identity evidence.\n"+
                    "<|vision_start|>"+"<|image_pad|>"*len(z)+"<|vision_end|>\n"+
                    "Score the usefulness of the directly visible local contents for the task. Background alone has low usefulness. Preserve historical identity evidence for unresolved hidden-state dependencies. Do not assign objects outside this local image to this patch. Return compact JSON with evidence first, then score (integer 0-100). Evidence is one sentence under 15 words: name the visible content and how it supports, or fails to support, the specific requested task. Do not repeat coordinates, general scene layout or instructions.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
                suffix=tokenizer(query,return_tensors="pt",add_special_tokens=False)["input_ids"].to(self.device)
                ids=torch.cat((ids0,suffix),dim=1)
                if ids.shape[1]+192>self.max_context:raise ValueError("complete patch request exceeds context")
                full_grid=torch.cat((all_grid,g))
                self.model.model.rope_deltas=None
                pos=self.model._prepare_position_ids_for_generation(ids,dict(attention_mask=torch.ones_like(ids),mm_token_type_ids=(ids==self.model.config.image_token_id).long(),image_grid_thw=full_grid))
                delta=self.model.model.rope_deltas.clone()
                grammar=build_transformers_prefix_allowed_tokens_fn(_DATA[key],JsonSchemaParser(dict(type="object",properties=dict(evidence=dict(type="string",maxLength=150),score=dict(type="integer",minimum=0,maximum=100)),required=["evidence","score"],additionalProperties=False)))
                kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=True,do_sample=False,max_new_tokens=192,logits_to_keep=1,eos_token_id=tokenizer.eos_token_id,prefix_allowed_tokens_fn=grammar)
                if request.get("fresh_branches"):
                    kwargs.update(position_ids=pos,mm_token_type_ids=(ids==self.model.config.image_token_id).long(),image_grid_thw=full_grid,mm_encoder_outputs=dict(image=BaseModelOutputWithPooling(pooler_output=tuple(features+[z]))))
                else:
                    # Consume the new image on a private branch with correct full
                    # multimodal positions, leaving the final text token for generate.
                    branch=self.model.model(input_ids=ids[:,length:-1],attention_mask=torch.ones_like(ids[:,:-1]),position_ids=pos[...,length:-1],past_key_values=branch_cache(base_cache),mm_encoder_outputs=dict(image=BaseModelOutputWithPooling(pooler_output=(z,))),image_grid_thw=g,use_cache=True)
                    kwargs["past_key_values"]=branch.past_key_values;del branch
                self.model.model.rope_deltas=delta
                generated=self.model.generate(**kwargs)
                if int(generated[0,-1])!=tokenizer.eos_token_id:raise ValueError("patch importance output truncated")
                raw=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip();item=json.loads(raw)
                if set(item)!={"evidence","score"} or type(item["score"]) is not int or not 0<=item["score"]<=100:raise ValueError("invalid patch importance")
                memo[identity]=item;del generated,kwargs
            item=memo[identity];result["visual_scores"].append(dict(id=c["id"],score=item["score"]))
            diagnostics.append(dict(**c,**item))
            if n%16==15:print(f"ROI_SCORE {n+1}/{len(candidates)} seconds={time.monotonic()-started:.1f}",flush=True)
        validate_scores(result,candidates)
        audit=dict(protocol=self.protocol,revision=self.scorer_revision,task=request["task"],current_time=request["current_time"],input_tokens=length,seconds=time.monotonic()-started,patches=diagnostics)
        with (self.directory/"roi_scores.jsonl").open("a") as f:f.write(json.dumps(audit)+"\n")
        print("ROI_COMPLETE",len(candidates),length,round(audit["seconds"],2),flush=True)
        return result

