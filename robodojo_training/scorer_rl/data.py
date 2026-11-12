"""Causal 25 Hz joint-action episodes, sampled head memory at 1 Hz."""
import json
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from opendm.data.transforms import Normalize, ChatTokenization, PadAction
from opendm.model.dm05.online_history import pack_history_features, encode_policy_history


def prepare_qwen(rows, encoder, output):
    output = Path(output);output.mkdir(parents=True,exist_ok=True)
    for row in rows:
        dest=output/(row["id"]+".pt")
        if dest.exists():
            saved=torch.load(dest,map_location="cpu",weights_only=True)
            if saved["revision"]!=encoder.revision or saved["source_sha256"]!=row["sha256"]:
                raise ValueError("Stale Qwen episode features")
            continue
        raw=np.load(row["path"])
        images=[Image.fromarray(im).resize((256,256),Image.Resampling.BICUBIC) for frame in raw["rgb"] for im in frame]
        with torch.no_grad():
            features=encoder.images(images).detach().cpu().half().reshape(len(raw["frames"]),3,64,-1)
            text=encoder.task(row["prompt"]).detach().cpu().half()
        tmp=dest.with_suffix(".tmp")
        torch.save(dict(features=features,text=text,revision=encoder.revision,source_sha256=row["sha256"]),tmp)
        tmp.replace(dest)
        print("QWEN_FEATURES_READY",row["id"],flush=True)


class Episode:
    def __init__(self,row,processor,model,feature_dir,revision,norm_path):
        self.row=row;self.raw=np.load(row["path"])
        self.frames=self.raw["frames"]; self.rgb=self.raw["rgb"]
        self.states=self.raw["states"]; self.actions=self.raw["actions"]
        if row["fps"]!=25 or self.actions.shape[1]!=14 or not np.array_equal(self.frames,np.arange(len(self.frames))*25):
            raise ValueError("Native DM05 data contract mismatch")
        self.qwen=torch.load(Path(feature_dir)/(row["id"]+".pt"),map_location="cpu",weights_only=True)
        if self.qwen["revision"]!=revision or self.qwen["source_sha256"]!=row["sha256"]:
            raise ValueError("Qwen features do not match raw RGB/revision")
        self.device=next(model.parameters()).device
        self.normalize=Normalize(norm_stats_path=norm_path,norm_keys=["state","action"],use_quantiles=True,clip_to_bounds=False)
        self.tokenize=ChatTokenization(processor,max_length=1536,image_prompts=["Head","Left wrist","Right wrist"],
                                      add_state=True,is_history=True,max_history_images=20,online_history=True)
        chunks=[]
        with torch.no_grad(), torch.autocast(self.device.type,dtype=torch.bfloat16,enabled=self.device.type=="cuda"):
            for start in range(0,len(self.frames),2):
                pixels=processor.image_processor(images=[Image.fromarray(im[0]) for im in self.rgb[start:start+2]],return_tensors="pt")["pixel_values"]
                chunks.append(encode_policy_history(model,pixels).detach().cpu())
        self.history=torch.cat(chunks)

    def sample(self,q):
        if not 1 <= q < len(self.frames):raise ValueError("Query outside causal episode")
        frame=int(self.frames[q]);start=max(0,q-20)
        raw=dict(prompt=self.row["prompt"], images=[Image.fromarray(im) for im in self.rgb[q]],history_images=[],
                 state=self.states[frame].copy(),action=self.actions[frame:frame+50].copy(),action_mask=np.ones((50,14),dtype=np.float32),
                 meta_data=dict(robot_type="Aloha",speed="0.5",fps=25,frame_index=frame))
        if raw["action"].shape!=(50,14):raise ValueError("Incomplete native action horizon")
        batch=PadAction(32)(self.tokenize(self.normalize(raw)))
        keep=("input_ids","attention_mask","token_type_ids","pixel_values","action","action_mask")
        sample={k:batch[k].to(self.device) for k in keep}
        sample["action"]=sample["action"].unsqueeze(0);sample["action_mask"]=sample["action_mask"].unsqueeze(0)
        features=self.history[start:q].to(self.device).flatten(0,1)
        ids,types,mask,features=pack_history_features(sample["input_ids"],sample["token_type_ids"],[features])
        sample.update(input_ids=ids,token_type_ids=types,history_mask=mask,history_features=features)
        patch_ids=torch.arange(start*16,q*16)
        times=(patch_ids//16).float()
        return sample,patch_ids,times

    def packet(self,q):
        # Current three views are context only; selectable keys are historical head KV.
        n=q+1;count=n*3*64
        xy=torch.stack(torch.meshgrid(torch.arange(8),torch.arange(8),indexing="ij"),-1).reshape(-1,2).float()/7
        return dict(features=self.qwen["features"][:n].flatten(0,2),text=self.qwen["text"],
                    times=torch.arange(n).float().repeat_interleave(192),xy=xy.repeat(n*3,1),
                    modalities=torch.zeros(count,dtype=torch.long),sensors=torch.arange(3).repeat_interleave(64).repeat(n),
                    query_time=float(q))


def native_logits(logits,candidate_ids):
    """Mean each head-camera 2x2 block of the scorer native 8x8 grid."""
    native=logits.reshape(-1,3,8,8)[:,0].reshape(-1,4,2,4,2).mean((2,4)).reshape(-1)
    ids=candidate_ids.to(logits.device)
    return native[ids]
