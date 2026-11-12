"""16-rank scorer GRPO; frozen native DM05 action FM, causal archive roll-in."""
import argparse,json,os,time,random,hashlib,gc
from datetime import timedelta
from pathlib import Path
import torch
import torch.distributed as dist
from training.parallel_scorer.model import load_scorer,save_scorer
from training.parallel_scorer.data import arguments
from training.parallel_scorer.rl import sample_modalities,path_logprob,group_loss,SensitivityWeights
from .data import prepare_qwen,Episode,native_logits
from .reward import FrozenFMReward


def atomic_json(path,value):
    path=Path(path);tmp=path.with_suffix(".tmp");tmp.write_text(json.dumps(value,indent=2));tmp.replace(path)


def signature(model):
    h=hashlib.sha256()
    for p in model.parameters():h.update(p.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def load_dm05(path,device):
    from opendm.model.dm05.dm05_arch import DM05Config,DM05ForConditionalGeneration
    from opendm.model.dm05.dm05_utils import configure_fp32_precision_runtime
    config=DM05Config.from_pretrained(path)
    config.vlm_config.vision_config._attn_implementation="sdpa"
    config.vlm_config.text_config._attn_implementation="sdpa"
    config.action_config._attn_implementation="sdpa"
    config.vision_attn_implementation="sdpa";config.llm_attn_implementation="sdpa";config.action_attn_implementation="sdpa"
    config.ikv_rgb_enabled=True;config.ikv_motion_only=False;config.chunk_size=50
    model=DM05ForConditionalGeneration.from_pretrained(path,config=config,dtype=torch.float32).to(device)
    model.set_precision_policy("fp32_mixed");configure_fp32_precision_runtime()
    model.set_attention_implementation(llm_attn_implementation="sdpa",vision_attn_implementation="sdpa",action_attn_implementation="sdpa",bf16=True)
    return model.eval().requires_grad_(False)


def main():
    p=argparse.ArgumentParser();p.add_argument("--config",required=True);a=p.parse_args()
    cfg=json.loads(Path(a.config).read_text());rank=int(os.environ.get("RANK",0));world=int(os.environ.get("WORLD_SIZE",1))
    local=int(os.environ.get("LOCAL_RANK",0));device=f"cuda:{local}";torch.cuda.set_device(local)
    if world>1:dist.init_process_group("nccl",timeout=timedelta(hours=2))
    out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=True)
    props=torch.cuda.get_device_properties(local)
    hardware=dict(rank=rank,local_rank=local,world_size=world,host=os.environ.get("HOSTNAME"),
                  visible_gpus=torch.cuda.device_count(),name=props.name,total_gib=props.total_memory/2**30,
                  uuid=str(getattr(props,"uuid","unavailable")),torch_version=torch.__version__)
    atomic_json(out/f"hardware-rank{rank}.json",hardware)
    print("HARDWARE",json.dumps(hardware),flush=True)
    torch.manual_seed(cfg["seed"]);model,info=load_scorer(cfg["checkpoint"],device)
    reference,_=load_scorer(cfg["checkpoint"],device);reference.eval().requires_grad_(False)
    rows=json.loads(Path(cfg["episodes"]).read_text())
    train=[r for r in rows if r["split"]=="train"];valid=[r for r in rows if r["split"]=="validation"]
    assert not ({r["episode_id"] for r in train}&{r["episode_id"] for r in valid})
    if len(train)<world:raise ValueError("Not enough independent training episodes")
    # Stable task-balanced assignment, no train/validation leakage.
    train=sorted(train,key=lambda r:(r["id"].rsplit("-",1)[1],r["task"]))
    assigned=train[rank::world]
    from training.parallel_scorer.encoders import QwenEncoders
    encoder=QwenEncoders(cfg["encoder"],info["encoder_revision"],device=device,image_batch=1)
    prepare_qwen(assigned,encoder,cfg["feature_dir"])
    del encoder;gc.collect();torch.cuda.empty_cache()
    atomic_json(out/f"features-rank{rank}.json",dict(rank=rank,episodes=[r["id"] for r in assigned]))
    if world>1:dist.barrier()
    from transformers import AutoProcessor
    processor=AutoProcessor.from_pretrained(cfg["dm05_checkpoint"],local_files_only=True)
    dm=load_dm05(cfg["dm05_checkpoint"],device);reward=FrozenFMReward(dm)
    optimizer=torch.optim.AdamW(model.parameters(),lr=cfg.get("learning_rate",1e-5))
    weights=SensitivityWeights(decay=.95,minimum=.25,maximum=4.)
    rng=random.Random(cfg["seed"]+rank);torch.manual_seed(cfg["seed"]+rank)
    episode_index=0;q=1;episode=None;accum=cfg.get("gradient_accumulation_frames",4)
    preflight=False;start_time=time.monotonic()
    if rank==0:atomic_json(out/"config.json",dict(cfg,world_size=world,train_episodes=len(train),validation_episodes=len(valid)))
    for update in range(1,cfg["max_updates"]+1):
        started=time.monotonic();pending=[];audits=[];optimizer.zero_grad(set_to_none=True)
        for _ in range(accum):
            if episode is None or q>=len(episode.frames):
                row=assigned[episode_index%len(assigned)];episode_index+=1;q=1
                episode=None;gc.collect()
                episode=Episode(row,processor,dm,cfg["feature_dir"],info["encoder_revision"],str(Path(cfg["dm05_checkpoint"])/"norm_stats.json"))
                reward.reset(row["prompt"])
                print("EPISODE",rank,episode_index,row["id"],len(episode.frames),flush=True)
            sample,ids,times=episode.sample(q)
            seed=rng.randrange(2**31);prepared=reward.prepare(sample,ids,times,seed=seed)
            candidates=prepared["candidates"];packet=episode.packet(q);args=arguments(packet,device)
            with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                model.eval();z=native_logits(model(**args).float(),candidates)
                ref=native_logits(reference(**args).float(),candidates)
            modes=torch.zeros(len(z),dtype=torch.long,device=device)
            # Budget is shared inside each group and strictly selective, including early history.
            budget=min(rng.choice(cfg["budgets"]),max(1,len(z)//2)) if len(z)<=min(cfg["budgets"]) else min(rng.choice(cfg["budgets"]),len(z)-1)
            with torch.no_grad():
                paths=[sample_modalities(z,modes,[budget,0]) for _ in range(4)]
                old=torch.stack([path_logprob(z,modes,path) for path in paths])
            losses=[reward.evaluate(prepared,candidates[path[0].cpu()]) for path in paths]
            if not preflight:
                native=reward.native_loss(prepared);all_kept=reward.evaluate(prepared,ids)
                with torch.no_grad(),torch.autocast("cuda",dtype=torch.bfloat16):
                    public_native=float(dm(**sample,fm_noise=prepared["noise"],fm_time=prepared["time"]).loss)
                if abs(native-public_native)>1e-5*max(1,abs(native)):raise RuntimeError("Public native FM parity failed")
                repeat=reward.evaluate(prepared,candidates[paths[0][0].cpu()])
                if abs(native-all_kept)>1e-5*max(1,abs(native)):raise RuntimeError("All-kept native FM parity failed")
                if repeat!=losses[0]:raise RuntimeError("Common-noise counterfactual is not deterministic")
                if reward.bank.generation!=0:raise RuntimeError("Candidate leaked into canonical archive")
                atomic_json(out/f"preflight-rank{rank}.json",dict(native=native,public_native=public_native,all_kept=all_kept,repeat_error=repeat-losses[0],losses=losses,
                    candidates=len(candidates),budget=budget,world=world,dm05_frozen=not any(p.requires_grad for p in dm.parameters())))
                preflight=True
            reward.commit_observation(prepared)
            pending.append((packet,candidates,modes,paths,old,ref,[-v for v in losses]))
            audits.append(dict(episode=episode.row["id"],query_frame=int(episode.frames[q]),query_seconds=q,budget=budget,
                               noise_seed=seed,time=float(prepared["time"][0]),losses=losses,candidates=len(candidates),
                               unique_sets=len({tuple(sorted(path[0].tolist())) for path in paths})))
            q+=1;del prepared,sample,z,args
        rewards=[v[-1] for v in pending]
        if world>1:
            all_groups=[None]*world;dist.all_gather_object(all_groups,rewards);all_rewards=[g for part in all_groups for g in part]
        else:all_rewards=rewards
        fractions,spread,raw=weights.weights(all_rewards);offset=rank*accum
        if update==1 and float(spread.max())<=1e-8:raise RuntimeError("Reward has no counterfactual sensitivity")
        for i,(packet,candidates,modes,paths,old,ref,rewards) in enumerate(pending):
            # Keep deterministic eval behavior for policy likelihoods; gradients remain enabled.
            model.eval()
            with torch.autocast("cuda",dtype=torch.bfloat16):new=native_logits(model(**arguments(packet,device)).float(),candidates)
            policy=group_loss(new,modes,paths,old,rewards)
            anchor=((new-new.mean())-(ref-ref.mean())).square().mean()
            loss=fractions[offset+i].to(device)*world*policy+cfg.get("reference_logit_weight",.01)*anchor/accum
            loss.backward();audits[i].update(policy_loss=float(policy.detach()),anchor=float(anchor.detach()),sensitivity=float(spread[offset+i]),query_weight=float(raw[offset+i]))
        if world>1:
            for parameter in model.parameters():
                if parameter.grad is None:parameter.grad=torch.zeros_like(parameter)
                dist.all_reduce(parameter.grad);parameter.grad.div_(world)
        norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True);optimizer.step()
        if update==1 or update%cfg.get("save_every",10)==0:
            hashes=[signature(model)]
            if world>1:
                dist.all_gather_object(hashes:=[None]*world,signature(model))
            if len(set(hashes))!=1:raise RuntimeError("Scorer replicas diverged")
            if rank==0:atomic_json(out/"SYNC_VERIFIED.json",dict(update=update,world_size=world,sha256=hashes[0]))
        atomic_json(out/f"rank{rank}-latest.json",dict(update=update,audits=audits,peak_gib=torch.cuda.max_memory_allocated()/2**30))
        if rank==0:
            summary=dict(update=update,reward_mean=float(torch.tensor(all_rewards).mean()),fm_action=float(-torch.tensor(all_rewards).mean()),
                         sensitivity_mean=float(spread.mean()),sensitivity_max=float(spread.max()),query_weight_min=float(raw.min()),query_weight_max=float(raw.max()),grad_norm=float(norm),seconds=time.monotonic()-started,
                         elapsed_seconds=time.monotonic()-start_time,queries=len(all_rewards),world_size=world,
                         peak_gpu_gib=torch.cuda.max_memory_allocated()/2**30)
            atomic_json(out/f"update-{update:06d}.json",summary)
            print(json.dumps(summary),flush=True)
            if update==1 or update%cfg.get("save_every",10)==0 or update==cfg["max_updates"]:
                tmp=out/f"scorer-{update}.tmp"
                save_scorer(model,tmp,encoder_revision=info["encoder_revision"],sensor_to_id=info.get("sensor_to_id"),
                            sensitivity_reference=weights.reference,dm05_revision=cfg["dm05_revision"])
                tmp.replace(out/f"scorer-{update}.pt")
                torch.save(dict(update=update,optimizer=optimizer.state_dict(),sensitivity_reference=weights.reference),out/f"optimizer-{update}.pt")
    if rank==0:atomic_json(out/"COMPLETE.json",dict(updates=cfg["max_updates"],seconds=time.monotonic()-start_time))
    if world>1:dist.destroy_process_group()

if __name__=="__main__":main()
