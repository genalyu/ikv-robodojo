"""Grounded independent scoring of native Sys1 KV patch positions.

Qwen localizes visual evidence once per immutable indexed frame. Geometric
intersection associates that evidence with each actual Sys1 patch. Full native
historical features and a task analysis condition independent Qwen scores.
Only vision features and localization records persist across requests.
"""
import copy,hashlib,json,time,math
from pathlib import Path
from unittest.mock import patch
import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling
from lmformatenforcer import JsonSchemaParser
from .qwen_sys2_service import QwenSys2 as Base
from .sys2 import validate_candidates,validate_scores,unwrap_json_fence
from .score_json import _DATA
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

def validate_grounding(regions):
    """Reject invalid/duplicated detections rather than silently alter evidence."""
    if not isinstance(regions,list) or len(regions)>32:
        raise ValueError("grounding must be a bounded list")
    boxes=set()
    for region in regions:
        if not isinstance(region,dict) or set(region)!={"label","bbox"}:
            raise ValueError("invalid grounding record")
        label=region["label"];box=region["bbox"]
        if not isinstance(label,str) or not label.strip() or len(label)>150:
            raise ValueError("invalid grounding label")
        if not isinstance(box,list) or len(box)!=4 or any(type(x) is not int or not 0<=x<=1000 for x in box) or not (box[0]<box[2] and box[1]<box[3]):
            raise ValueError("invalid grounding box")
        # Identical boxes with different labels still contaminate the same patch.
        key=tuple(box)
        if key in boxes:raise ValueError("duplicate grounding box")
        boxes.add(key)

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
    protocol="grounded_independent_v1"
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.groundings={}
        self.patch_prompt=Path(__file__).with_name("patch_score_prompt.txt").read_text()
        self.ground_prompt=Path(__file__).with_name("grounding_prompt.txt").read_text()
        self.scorer_revision=hashlib.sha256(Path(__file__).read_bytes()+self.patch_prompt.encode()+self.ground_prompt.encode()).hexdigest()

    def encode(self,encoded_image):
        result=super().encode(encoded_image)
        self.ground(result["frame"])
        return result

    @torch.inference_mode()
    def ground(self, frame):
        if frame in self.groundings:
            return self.groundings[frame]
        question=self.ground_prompt
        settings=dict(version=5,repetition_penalty=1.1,max_new_tokens=2048,retry_temperature=0.7,retry_top_p=0.8,retry_top_k=20)
        signature=hashlib.sha256((self.provenance+question+json.dumps(settings,sort_keys=True)).encode()).hexdigest()
        path=self.directory/(str(frame)+".grounding-v5.json")
        if path.exists():
            record=json.loads(path.read_text())
            if record["signature"]==signature:
                validate_grounding(record["regions"])
                self.groundings[frame]=record["regions"]
                return record["regions"]
        packet=torch.load(self.directory/(str(frame)+".pt"),map_location="cpu",weights_only=True)
        if packet["provenance"]!=self.provenance:raise ValueError("grounding feature provenance mismatch")
        feature=packet["features"].to(self.device)
        content=[dict(type="text",text="<|vision_start|>"+"<|image_pad|>"*len(feature)+"<|vision_end|>"),dict(type="text",text=question)]
        messages=[dict(role="system",content="You are a precise visual grounding assistant. Report only visible evidence."),dict(role="user",content=content)]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.processor.tokenizer(text,return_tensors="pt",add_special_tokens=False).to(self.device)
        inputs["mm_token_type_ids"]=(inputs["input_ids"]==self.model.config.image_token_id).long()
        inputs["image_grid_thw"]=packet["grid"].to(self.device)
        inputs["mm_encoder_outputs"]=dict(image=BaseModelOutputWithPooling(pooler_output=(feature,)))
        key=id(self.processor.tokenizer)
        if key not in _DATA:_DATA[key]=build_token_enforcer_tokenizer_data(self.processor.tokenizer)
        native_schema={"type":"array","maxItems":32,"items":{"type":"object","properties":{"bbox_2d":{"type":"array","items":{"type":"integer","minimum":0,"maximum":1000},"minItems":4,"maxItems":4},"label":{"type":"string","maxLength":150}},"required":["bbox_2d","label"],"additionalProperties":False}}
        def parse_grounding(raw):
            values=json.loads(unwrap_json_fence(raw))
            if not isinstance(values,list) or len(values)>32:raise ValueError("grounding must be a bounded JSON list")
            regions=[]
            for v in values:
                if set(v)!={"label","bbox_2d"} or not isinstance(v["label"],str) or not v["label"].strip():raise ValueError("invalid grounding label")
                box=v["bbox_2d"]
                if not isinstance(box,list) or len(box)!=4 or any(type(x) is not int or not 0<=x<=1000 for x in box) or not (box[0]<box[2] and box[1]<box[3]):raise ValueError("invalid grounding box")
                regions.append(dict(label=v["label"],bbox=box))
            validate_grounding(regions)
            return dict(regions=regions)
        for attempt in range(3):
            self.model.model.rope_deltas=None
            enforced=build_transformers_prefix_allowed_tokens_fn(_DATA[key],JsonSchemaParser(native_schema)) if attempt==2 else None
            # Greedy failures repeat even with a JSON grammar. A frame-keyed seed
            # gives a reproducible alternative without affecting other requests' RNG.
            sampling=dict(do_sample=True,temperature=0.7,top_p=0.8,top_k=20) if attempt else dict(do_sample=False)
            with torch.random.fork_rng(devices=[self.device] if self.device.type=="cuda" else []):
                torch.manual_seed(frame % (2**31))
                generated=self.model.generate(**inputs,max_new_tokens=2048,**sampling,repetition_penalty=1.1,use_cache=True,past_key_values=None,logits_to_keep=1,eos_token_id=self.processor.tokenizer.eos_token_id,prefix_allowed_tokens_fn=enforced)
            raw=self.processor.tokenizer.decode(generated[0,inputs["input_ids"].shape[1]:],skip_special_tokens=True).strip()
            try:
                if int(generated[0,-1])!=self.processor.tokenizer.eos_token_id:raise ValueError("grounding output truncated")
                result=parse_grounding(raw)
                break
            except (ValueError,KeyError,TypeError) as exc:
                self.directory.joinpath(str(frame)+f".ground-v5-invalid-{attempt}.json").write_text(json.dumps(dict(attempt=attempt,response=raw,error=str(exc))))
                if attempt==2:raise ValueError(f"Qwen grounding failed all validation attempts for frame {frame}: {exc}") from exc
        record=dict(signature=signature,frame=frame,regions=result["regions"],attempt=attempt)
        temporary=path.with_suffix(".tmp");temporary.write_text(json.dumps(record));temporary.replace(path)
        print("GROUNDING",json.dumps(record),flush=True)
        self.groundings[frame]=result["regions"]
        return result["regions"]

    @torch.inference_mode()
    def score(self,request):
        request_started=time.monotonic()
        candidates=validate_candidates(request["candidates"])
        if not candidates:return dict(visual_scores=[],tactile_scores=[])
        if any(c["modality"]!="visual" for c in candidates):raise ValueError("RoboDojo scorer expects visual candidates")
        if not isinstance(request.get("task"),str) or not request["task"].strip():raise ValueError("empty task")
        context=request.get("context_index",[dict(frame=c["frame"],time=c["time"],modality=c["modality"],grid=c["grid"]) for c in candidates])
        frames=request.get("context_frames",list(dict.fromkeys(c["frame"] for c in candidates)))
        if len(set(frames))!=len(frames) or not set(c["frame"] for c in candidates)<=set(frames):raise ValueError("invalid context")
        frame_times=observation_times(frames,context,candidates)
        frame_time={f:frame_times[f][0] for f in frames}
        frames=sorted(frames,key=lambda f:(frame_time[f],f));aliases={f:i for i,f in enumerate(frames)}
        grounded={f:self.ground(f) for f in frames}
        content=[];features=[];grids=[]
        for f in frames:
            packet=torch.load(self.directory/(str(f)+".pt"),map_location="cpu",weights_only=True)
            if packet["provenance"]!=self.provenance:raise ValueError("feature provenance mismatch")
            features.append(packet["features"].to(self.device));grids.append(packet["grid"])
            content.extend([dict(type="text",text=f"Frame {aliases[f]}, observed at times {json.dumps(frame_times[f],separators=(',',':'))}:"),dict(type="text",text="<|vision_start|>"+"<|image_pad|>"*len(features[-1])+"<|vision_end|>")])
        content.append(dict(type="text",text="Observed localized regions by frame: "+json.dumps([dict(frame=aliases[f],regions=grounded[f]) for f in frames],separators=(",",":"))))
        system=self.patch_prompt
        content.append(dict(type="text",text=f"Task: {request['task']}\nCurrent time: {request['current_time']}\n"))
        messages=[dict(role="system",content=system),dict(role="user",content=content)]
        planning_messages=[messages[0],dict(role="user",content=content+[dict(type="text",text="Before patch scoring, summarize the task-relevant observed scene in at most 80 words. Identify the target and non-targets, their locations, the historical evidence needed for identity/order, and the current observed state relevant to the next operation. Use the frame labels and object coordinates to avoid left/right confusion. Do not infer completed actions from instructions. State uncertainty when a binding is unknown. Do not output scores.")])]
        ptext=self.processor.apply_chat_template(planning_messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        pin=self.processor.tokenizer(ptext,return_tensors="pt",add_special_tokens=False).to(self.device)
        if pin["input_ids"].shape[1]+512>self.max_context:raise ValueError("complete planning context exceeds limit")
        self.model.model.rope_deltas=None
        plan_out=self.model.generate(**pin,mm_token_type_ids=(pin["input_ids"]==self.model.config.image_token_id).long(),image_grid_thw=torch.cat(grids).to(self.device),mm_encoder_outputs=dict(image=BaseModelOutputWithPooling(pooler_output=tuple(features))),use_cache=True,do_sample=False,max_new_tokens=512,logits_to_keep=1,eos_token_id=self.processor.tokenizer.eos_token_id)
        if int(plan_out[0,-1])!=self.processor.tokenizer.eos_token_id:raise ValueError("task analysis output was truncated")
        task_plan=self.processor.tokenizer.decode(plan_out[0,pin["input_ids"].shape[1]:],skip_special_tokens=True).strip()
        del plan_out,pin
        print("TASK_ANALYSIS",task_plan,flush=True)
        messages=[planning_messages[0],planning_messages[1],dict(role="assistant",content=task_plan),dict(role="user",content=[dict(type="text",text="Now assess one localized patch against that task analysis.\n")])]
        rendered=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=False,enable_thinking=False)
        assert rendered.endswith("<|im_end|>\n")
        prefix=rendered[:-len("<|im_end|>\n")]
        tokenizer=self.processor.tokenizer
        prefix_ids=tokenizer(prefix,return_tensors="pt",add_special_tokens=False)["input_ids"].to(self.device)
        length=prefix_ids.shape[1]
        if length+1024>self.max_context:raise ValueError("complete history exceeds context")
        mask=prefix_ids==self.model.config.image_token_id
        if int(mask.sum())!=sum(len(x) for x in features):raise ValueError("feature/placeholder mismatch")
        native=dict(image=BaseModelOutputWithPooling(pooler_output=tuple(features)))
        inputs=dict(input_ids=prefix_ids,attention_mask=torch.ones_like(prefix_ids),mm_token_type_ids=mask.long(),image_grid_thw=torch.cat(grids).to(self.device),mm_encoder_outputs=native,use_cache=True)
        started=time.monotonic();self.model.model.rope_deltas=None
        output=self.model.model(**inputs)
        base_cache=output.past_key_values;deltas=self.model.model.rope_deltas.clone()
        del output
        result=dict(visual_scores=[],tactile_scores=[])
        key=id(tokenizer)
        if key not in _DATA:_DATA[key]=build_token_enforcer_tokenizer_data(tokenizer)
        diagnostics=[]
        query_results={}
        for index,c in enumerate(sorted(candidates,key=lambda c:(frame_time[c["frame"]],c["frame"],c["time"],tuple(c["grid"]),c["patch"],c["id"]))):
            h,w=c["grid"];row,col=divmod(c["patch"],w)
            box=[round(col*1000/w),round(row*1000/h),round((col+1)*1000/w),round((row+1)*1000/h)]
            local=[]
            for region in grounded[c["frame"]]:
                x0,y0,x1,y1=region["bbox"]
                ix0,iy0,ix1,iy1=max(x0,box[0]),max(y0,box[1]),min(x1,box[2]),min(y1,box[3])
                if ix1>ix0 and iy1>iy0:
                    local.append(dict(label=region["label"],object_box=region["bbox"],patch_coverage_percent=round(100*(ix1-ix0)*(iy1-iy0)/((box[2]-box[0])*(box[3]-box[1])),1)))
            query=f"""Assess ONE patch from Frame {aliases[c["frame"]]}, observation time {c["time"]}.
The spatial index has already computed which detected regions intersect this patch:
{json.dumps(local,separators=(",",":"))}
These are the localized contents of this patch; object_box locates the associated object in the full image for historical identity matching. A patch may show only part of an object. Do not require the whole object to fit inside a patch. Judge the task utility of this local evidence, using all historical context to distinguish target/non-target and current/past states.
Return compact JSON with a short evidence phrase and integer score 0-100.<|im_end|>
<|im_start|>assistant
<think>

</think>

"""
            if query not in query_results:
                suffix=tokenizer(query,return_tensors="pt",add_special_tokens=False)["input_ids"].to(self.device)
                ids=torch.cat([prefix_ids,suffix],dim=1)
                if ids.shape[1]+128>self.max_context:raise ValueError("complete patch query exceeds context")
                grammar=build_transformers_prefix_allowed_tokens_fn(_DATA[key],JsonSchemaParser({"type":"object","properties":{"evidence":{"type":"string","maxLength":100},"score":{"type":"integer","minimum":0,"maximum":100}},"required":["evidence","score"],"additionalProperties":False}))
                self.model.model.rope_deltas=deltas.clone()
                kwargs=dict(input_ids=ids,attention_mask=torch.ones_like(ids),past_key_values=branch_cache(base_cache),use_cache=True,do_sample=False,max_new_tokens=128,logits_to_keep=1,eos_token_id=tokenizer.eos_token_id,prefix_allowed_tokens_fn=grammar)
                if request.get("fresh_branches"):
                    kwargs.pop("past_key_values")
                    kwargs.update(mm_token_type_ids=(ids==self.model.config.image_token_id).long(),image_grid_thw=inputs["image_grid_thw"],mm_encoder_outputs=native)
                    self.model.model.rope_deltas=None
                generated=self.model.generate(**kwargs)
                raw=tokenizer.decode(generated[0,ids.shape[1]:],skip_special_tokens=True).strip()
                item=json.loads(raw)
                if set(item)!={"evidence","score"} or not isinstance(item["evidence"],str) or type(item["score"]) is not int or not 0<=item["score"]<=100:raise ValueError("invalid grounded importance")
                query_results[query]=item
                del kwargs,generated
            item=query_results[query]
            raw=str(item["score"])
            result["visual_scores"].append(dict(id=c["id"],score=int(raw)))
            diagnostics.append(dict(id=c["id"],frame=c["frame"],patch=c["patch"],time=c["time"],score=int(raw),evidence=item["evidence"],localized_evidence=local))
            if index%16==15:print(f"PATCH_SCORE {index+1}/{len(candidates)} seconds={time.monotonic()-started:.1f}",flush=True)
        validate_scores(result,candidates)
        elapsed=time.monotonic()-started
        print(f"INDEPENDENT_COMPLETE patches={len(candidates)} frames={len(frames)} tokens={length} seconds={elapsed:.2f}",flush=True)
        with (self.directory/"patch_scores_grounded_v1.jsonl").open("a") as f:f.write(json.dumps(dict(protocol=self.protocol,revision=self.scorer_revision,unique_patch_queries=len(query_results),context_frames=frames,observation_times=frame_times,total_seconds=time.monotonic()-request_started,task=request["task"],current_time=request["current_time"],input_tokens=length,seconds=elapsed,task_analysis=task_plan,patches=diagnostics))+"\n")
        return result
