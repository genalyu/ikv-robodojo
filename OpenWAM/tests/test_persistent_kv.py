import pytest
import torch
from openwam.model.persistent_kv import PersistentKVBank

def test_refresh_never_resurrects_and_keeps_unique_old_information():
    bank = PersistentKVBank(16)
    evicted = set()
    previous = set()
    for step in range(35):
        # Refresh overlapping window, including many already evicted rows.
        ids = torch.arange(max(0,step-19)*16,(step+1)*16)
        d = torch.zeros(len(ids),2); d[:,1] = 1; d[ids==0] = torch.tensor([1.,0.])
        p = bank.plan(ids,d,ids//16)
        payload = ids.float()[None,:,None].requires_grad_()
        k,v = p.merge(0,payload,payload+1,1)
        selected = set(p.metadata['ids'].tolist())
        assert len(selected)==16 and not selected & evicted
        assert k.flatten().tolist()==p.metadata['ids'].float().tolist()
        candidates = previous | set(range(step*16,(step+1)*16))
        evicted |= candidates-selected
        bank.commit(p,{0:(k,v)},1)
        previous = selected
        if step == 21:
            assert 0 in previous  # Older than the 20-frame refresh window.
    assert not bank.layers[0][0].requires_grad
    assert not hasattr(bank,'archive')

def test_transaction_failure_and_current_gradients():
    b = PersistentKVBank(4)
    p = b.plan(torch.arange(4),torch.eye(4),torch.zeros(4))
    x = torch.randn(1,4,2,requires_grad=True)
    k,v = p.merge(0,x,x,1)
    (k+v).sum().backward()
    assert x.grad.abs().sum()>0
    with pytest.raises(RuntimeError,match='incomplete'):
        b.commit(p,{0:(k,v)},2)
    assert b.generation==0 and b.metadata is None
    b.commit(p,{0:(k,v)},1)
    with pytest.raises(RuntimeError,match='stale'):
        b.commit(p,{0:(k,v)},1)


def test_joint_read_budget_first_frame_parity_and_motion_independence():
    import numpy as np
    from types import SimpleNamespace
    from openwam.model.persistent_kv import OpenWAMKVSession
    from torch.nn.functional import scaled_dot_product_attention as sdpa
    def attn(q,k,v,m):
        return sdpa(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),m).transpose(1,2)
    driver = SimpleNamespace(_mixed_attention=attn)
    holder = {}
    d = torch.zeros(1,2,2,2); d[...,1] = 1; d[0,0,0] = torch.tensor([1.,0.])
    image = np.zeros((8,8,3),dtype=np.uint8)
    q,k,v = [torch.randn(1,9,2,4,requires_grad=True) for _ in range(3)]
    mask = torch.ones(9,9,dtype=torch.bool); mask[:4,4:] = False
    session = OpenWAMKVSession(holder,d,image)
    session.prepare(2,2,'cpu')
    result = session.attention(driver,0,q,k,v,mask)
    torch.testing.assert_close(result,attn(q,k,v,mask))
    result.sum().backward()
    assert k.grad.abs().sum()>0
    session.commit(1)
    old = holder['bank'].layers[0][0].clone()
    d[...,0]=0; d[...,1]=1
    second = OpenWAMKVSession(holder,d,image)
    second.prepare(2,2,'cpu')
    assert 0 in second.plan.metadata['ids']
    calls = []
    def capture(q,k,v,m):
        calls.append(k.shape[1]); return attn(q,k,v,m)
    second.attention(SimpleNamespace(_mixed_attention=capture),0,q,k*2,v,mask)
    assert calls==[4,9]  # Clean read N; action/future read N plus five suffix tokens.
    assert torch.equal(holder['bank'].layers[0][0],old)  # No commit on failed generation.
    gated = OpenWAMKVSession(holder,d,image,motion_only=True)
    gated.prepare(2,2,'cpu')
    assert torch.equal(gated.plan.metadata['ids'],second.plan.metadata['ids'])
    second.commit(1)
    assert holder['bank'].layers[0][0].shape[1]==4


def test_real_wan_joint_forward_clean_cache_invariant_and_backward():
    import numpy as np
    from types import SimpleNamespace
    from openwam.model.video_backbone.wan.models.dit import WanModel
    from openwam.model.video_backbone.wan_backbone import WanBase
    from openwam.model import build_architecture
    from openwam.model.persistent_kv import OpenWAMKVSession
    dit = WanModel(dim=32,in_dim=4,ffn_dim=64,out_dim=4,text_dim=16,freq_dim=16,eps=1e-6,patch_size=(1,2,2),num_heads=4,num_layers=2,has_image_input=False,seperated_timestep=True,require_vae_embedding=False,require_clip_embedding=False,fuse_vae_embedding_in_latents=True)
    class TinyWan(WanBase):
        @classmethod
        def from_pretrained(cls,*args,**kwargs):
            raise NotImplementedError
    vb = TinyWan(SimpleNamespace(dit=dit))
    vb._device = torch.device('cpu'); vb._dtype = torch.float32
    arch = build_architecture('dual_system_self_attn',dict(framework='dual_system',variant='joint_self_attn',video_dim=32,num_dit_layers=2,bridge_interval=1,dim=32,ffn_dim=64,num_heads=4,action_dim=4,text_dim=16,use_proprioception=False,attention_mask_mode='mutual',video_attention_mask_mode='first_frame_causal'))
    arch.video_backbone = vb
    arch.build_mot_driver()
    context = torch.randn(1,3,16)
    latent = torch.randn(1,4,3,4,4)
    session = OpenWAMKVSession({},torch.randn(1,2,2,8),np.zeros((8,8,3),dtype=np.uint8))
    kw = dict(latents=latent,timestep=torch.tensor([500.]),context=context,context_mask=torch.ones(1,3,dtype=torch.bool),first_frame_latents=latent[:,:,:1],num_clean_prefix_frames=1,fuse_vae_embedding_in_latents=True,ikv_kv_session=session)
    video,action = arch(noisy_actions=torch.randn(1,2,4),action_timestep=torch.tensor([500.]),**kw)
    captured = {i:(k.detach().clone(),v.detach().clone()) for i,(k,v) in session.layers.items()}
    changed = latent.clone(); changed[:,:,1:] = torch.randn_like(changed[:,:,1:])
    kw.update(latents=changed,timestep=torch.tensor([100.]))
    video,action = arch(noisy_actions=torch.randn(1,2,4)*3,action_timestep=torch.tensor([100.]),use_gradient_checkpointing=True,**kw)
    for i,(k,v) in session.layers.items():
        torch.testing.assert_close(k,captured[i][0])
        torch.testing.assert_close(v,captured[i][1])
    (video.square().mean()+action.square().mean()).backward()
    assert dit.patch_embedding.weight.grad is not None and dit.patch_embedding.weight.grad.abs().sum()>0
    session.commit(2)
    assert session.bank.capacity==4

    from openwam.model.streaming_training import train_episode
    from tests.test_openwam_trainer import _MockScheduler
    vb._scheduler=_MockScheduler()
    arch.action_backbone.scheduler=_MockScheduler()
    arch._device=torch.device('cpu'); arch._dtype=torch.float32
    steps=[]
    for t in range(2):
        target=torch.randn(1,4,3,4,4)
        steps.append(dict(input_latents=target,first_frame_latents=target[:,:,:1],context=context.detach(),context_mask=torch.ones(1,3,dtype=torch.bool),fuse_vae_embedding_in_latents=True,actions=torch.randn(1,2,4),ikv_current_image=np.zeros((8,8,3),dtype=np.uint8),ikv_dino_features=torch.randn(1,2,2,8)))
    optimizer=torch.optim.AdamW(arch.parameters(),lr=1e-3)
    before=dit.patch_embedding.weight.detach().clone()
    loss=train_episode(arch,steps,optimizer)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before,dit.patch_embedding.weight)
