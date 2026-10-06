"""Ordered joint video/action KV training; uses deployment's exact bank and attention."""
import torch
from .persistent_kv import OpenWAMKVSession


def train_episode(architecture, steps, optimizer, *, motion_only=False):
    """Steps are prepare_inputs outputs plus ikv_current_image/ikv_dino_features.

    Each input_latents starts with ONE actual observed frame, followed by future
    target video. Never pass a historical clip as the clean prefix. The supplied
    DINO grid describes the current composite view and is detached by selection.
    Update weights after the whole ordered episode; detach bank between steps.
    """
    if not steps:
        raise ValueError('empty episode')
    architecture.train()
    state={}
    optimizer.zero_grad(set_to_none=True)
    losses=[]
    for step in steps:
        inputs=dict(step)
        image=inputs.pop('ikv_current_image')
        descriptors=inputs.pop('ikv_dino_features')
        session=OpenWAMKVSession(state,descriptors,image,motion_only=motion_only)
        ref=inputs.get('first_frame_latents')
        if ref is None or ref.shape[2]!=1:
            raise ValueError('streaming joint training requires exactly one observed latent frame')
        inputs.update(ikv_kv_session=session,num_clean_prefix_frames=1,zero_clean_prefix_t_mod=True)
        out=architecture.compute_loss(**inputs)
        if not torch.isfinite(out['loss']):
            raise RuntimeError('nonfinite streaming loss')
        (out['loss']/len(steps)).backward()
        session.commit(architecture.video_backbone.num_layers)
        losses.append(float(out['loss'].detach()))
    torch.nn.utils.clip_grad_norm_([p for p in architecture.parameters() if p.requires_grad],1.,error_if_nonfinite=True)
    optimizer.step()
    return sum(losses)/len(losses)
