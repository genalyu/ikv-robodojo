"""Episode-ordered, detached recurrent KV training using the deployment splice path."""
import torch
from .persistent_kv import PersistentKVBank
from .qwen_archive import QwenArchiveBank
from .online_history import encode_policy_history, pack_history_features
from .ikv_dino import encode_dino_grid
from .ikv_rgb import history_motion_scores
from .history_update import select_history_update, validate_update_mode


def prepare_stream_step(model, step, bank, *, update_mode="window_refresh"):
    """A step contains the actual current multimodal prefix and <=20 history frames.

    history_patch_ids are absolute episode frame_index*16+patch_index, not
    window-relative IDs. history_times are seconds, repeated 16 times per frame.
    Raw RGB/pixels must follow those IDs. DINO is frozen; vision and KV keep grads.
    """
    data = select_history_update(step, bank, update_mode)
    ids = data.pop('history_patch_ids')
    times = data.pop('history_times')
    descriptors = data.pop('history_descriptors',None)
    features = data.get('history_features')
    pixels = data.get('history_pixel_values')
    rgb = data.get('history_rgb_values')
    if len(ids)>bank.capacity or len(ids)%16:
        raise ValueError('stream step must contain <=20 complete history frames')
    if features is None:
        features = encode_policy_history(model,pixels).flatten(0,1) if len(ids) else next(model.parameters()).new_empty((0,model.model.vlm.model.language_model.config.hidden_size))
    if descriptors is None:
        if len(ids) and isinstance(bank, QwenArchiveBank):
            raise ValueError("archive banks require explicit history_descriptors: frame IDs for Qwen, DINO plus relevance for Formula")
        if len(ids):
            descriptors = encode_dino_grid(rgb,checkpoint=model.config.ikv_dino_model_path,grid=(4,4),device=features.device).reshape(len(ids),-1)
        else:
            old = getattr(bank, "archive_metadata", None)
            if old is None:
                old = getattr(bank, "metadata", None)
            width = old["d"].shape[-1] if old is not None and "d" in old else 1
            if old is not None and "q" in old:
                width += 1
            descriptors = features.new_empty((0,width))
    packed,types,mask,features = pack_history_features(data['input_ids'],data['token_type_ids'],[features])
    data.update(input_ids=packed,token_type_ids=types,history_mask=mask,history_features=features)
    data.pop('history_frame_counts',None)
    motion = data.pop('history_motion',None)
    if motion is None and rgb is not None and len(ids):
        motion = history_motion_scores(rgb).flatten()
    data['ikv_kv_request'] = dict(bank=bank,ids=ids,descriptors=descriptors,times=times,motion=motion,contact=data.pop('history_contact',None),motion_only=model.config.ikv_motion_only,motion_threshold=model.config.ikv_motion_threshold)
    query_time = data.pop("history_query_time", None)
    if query_time is not None:
        data["ikv_kv_request"]["query_time"] = query_time
    return data


def train_episode(model, steps, optimizer, *, update_mode="window_refresh"):
    """One optimizer update per ordered episode; previous KV is detached.

    Backward each step immediately to bound activation memory. Reset the bank at
    each episode and each optimizer update, so cached KV never crosses weights.
    Each step must carry its own actual prompt/state/current cameras; rebuilding
    old steps using the final observation would leak future information.
    """
    validate_update_mode(update_mode)
    if not steps:
        raise ValueError('empty episode')
    model.train()
    bank = PersistentKVBank(model.config.ikv_history_capacity)
    optimizer.zero_grad(set_to_none=True)
    losses=[]
    for step in steps:
        prepared = prepare_stream_step(model,step,bank,update_mode=update_mode)
        out = model(**prepared)
        (out.loss/len(steps)).backward()
        losses.append(float(out.loss.detach()))
    torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.0,error_if_nonfinite=True)
    optimizer.step()
    return sum(losses)/len(losses)
