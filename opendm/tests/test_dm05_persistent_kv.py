import pytest
import torch
from opendm.model.dm05.persistent_kv import PersistentKVBank

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


def test_real_dm05_stream_train_gradients_sliding_layers_and_inference():
    from transformers import Gemma3Config,Gemma3TextConfig,SiglipVisionConfig
    from opendm.model.dm05.dm05_arch import DM05Config,DM05ForConditionalGeneration
    from opendm.model.dm05.online_history import pack_history_features
    text = Gemma3TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,layer_types=['sliding_attention','full_attention'],sliding_window=16,max_position_embeddings=1024)
    vision = SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,image_size=16,patch_size=4)
    cfg = DM05Config(vlm_config=Gemma3Config(text_config=text.to_dict(),vision_config=vision.to_dict(),mm_tokens_per_image=16,image_token_index=127),action_config=text,action_dim=4,chunk_size=2,ikv_rgb_enabled=True)
    model = DM05ForConditionalGeneration(cfg)
    bank = PersistentKVBank(320)
    base = torch.tensor([[2]+[7]*320+[3]])
    for step in range(3):
        # Start empty; then fill window; then refresh overlap plus new patches.
        start,end = [(0,0),(0,320),(16,336)][step]
        features = torch.randn(end-start,32,requires_grad=True)
        ids,types,hm,features_packed = pack_history_features(base,torch.zeros_like(base),[features])
        patch_ids = torch.arange(start,end)
        descriptors = torch.ones(len(patch_ids),2)
        request = dict(bank=bank,ids=patch_ids,descriptors=descriptors,times=patch_ids//16)
        kw = dict(input_ids=ids,attention_mask=torch.ones_like(ids),token_type_ids=types,history_mask=hm,history_features=features_packed,ikv_kv_request=request)
        model.train()
        out = model(**kw,action=torch.randn(1,2,4),action_mask=torch.ones(1,2,4))
        out.loss.backward()
        if len(features):
            assert features.grad is not None and features.grad.abs().sum()>0
        assert bank.generation==step+1
        assert all(k.shape[-2]==min(end,320) for k,v in bank.layers.values())
        assert all(not k.requires_grad for k,v in bank.layers.values())
        model.zero_grad(set_to_none=True)
    model.eval()
    result = model.inference_action(**kw,diffusion_steps=2,action_mask=torch.ones(1,1,4))
    assert torch.isfinite(result).all() and result.shape==(1,2,4)
    assert bank.generation==4

    from opendm.model.dm05.streaming_training import train_episode
    steps=[]
    for t in range(2):
        patch_ids=torch.arange(t*16,(t+1)*16)
        steps.append(dict(input_ids=base,attention_mask=torch.ones_like(base),token_type_ids=torch.zeros_like(base),history_features=torch.randn(16,32),history_patch_ids=patch_ids,history_times=torch.full((16,),float(t)),history_descriptors=torch.ones(16,2),action=torch.randn(1,2,4),action_mask=torch.ones(1,2,4)))
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
    before=model.model.action_out_proj.weight.detach().clone()
    loss=train_episode(model,steps,optimizer)
    assert torch.isfinite(torch.tensor(loss))
    assert not torch.equal(before,model.model.action_out_proj.weight)
