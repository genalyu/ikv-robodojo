import torch
import pytest
from transformers import Gemma3Config,Gemma3TextConfig,SiglipVisionConfig
from opendm.model.dm05.dm05_arch import DM05Config,DM05ForConditionalGeneration
from opendm.model.dm05.online_history import pack_history_features
from robodojo_training.scorer_rl.reward import FrozenFMReward
from robodojo_training.scorer_rl.data import native_logits


def tiny():
    torch.manual_seed(17)
    text=Gemma3TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,
                         num_key_value_heads=2,head_dim=8,layer_types=["sliding_attention","full_attention"],sliding_window=16,max_position_embeddings=1024)
    vision=SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,image_size=16,patch_size=4)
    cfg=DM05Config(vlm_config=Gemma3Config(text_config=text.to_dict(),vision_config=vision.to_dict(),mm_tokens_per_image=16,image_token_index=127),
                   action_config=text,action_dim=4,chunk_size=2,ikv_rgb_enabled=True)
    return DM05ForConditionalGeneration(cfg).eval()


def sample(start=0,end=32):
    base=torch.tensor([[2]+[7]*320+[3]])
    ids,types,mask,features=pack_history_features(base,torch.zeros_like(base),[torch.randn(end-start,32)])
    return dict(input_ids=ids,token_type_ids=types,attention_mask=torch.ones_like(ids),history_mask=mask,
                history_features=features,action=torch.randn(1,2,4),action_mask=torch.ones(1,2,4))


def test_native_fm_shared_draws_subset_sensitivity_and_transaction_isolation():
    m=tiny();r=FrozenFMReward(m);r.reset("memory")
    s=sample();ids=torch.arange(32);p=r.prepare(s,ids,ids//16,seed=42)
    expected=r.native_loss(p);actual=r.evaluate(p,ids)
    assert actual==pytest.approx(expected,abs=1e-7)
    a=r.evaluate(p,ids[:16]);b=r.evaluate(p,ids[16:])
    assert a!=b
    assert r.evaluate(p,ids[:16])==a
    assert r.bank.generation==0 and r.bank.archive_metadata is None
    # The public native forward consumes the exact same explicit FM target.
    with torch.no_grad():out=m(**s,fm_noise=p["noise"],fm_time=p["time"])
    assert float(out.loss)==pytest.approx(expected,abs=1e-7)
    r.commit_observation(p)
    assert r.bank.generation==1 and torch.equal(r.bank.archive_metadata["ids"],ids)
    oldkeys={i:k.clone() for i,(k,v) in r.bank.archive_layers.items()}
    s2=sample(16,48);ids2=torch.arange(16,48);p2=r.prepare(s2,ids2,ids2//16,seed=43)
    assert p2["candidates"].tolist()==list(range(48))
    r.evaluate(p2,torch.arange(16))
    assert r.bank.generation==1
    for i,k in oldkeys.items():assert torch.equal(r.bank.archive_layers[i][0],k)
    with pytest.raises(ValueError):r.evaluate(p2,torch.tensor([99]))
    assert all(x.grad is None and not x.requires_grad for x in m.parameters())


def test_native_logit_pooling_keeps_head_only_and_backpropagates():
    z=torch.arange(384.).requires_grad_();ids=torch.tensor([0,15,16,31])
    result=native_logits(z,ids)
    torch.testing.assert_close(result,torch.tensor([4.5,58.5,196.5,250.5]))
    result.sum().backward()
    assert z.grad.nonzero().numel()==16
    assert not z.grad[64:192].any()


def test_empty_action_mask_is_rejected():
    m=tiny();s=sample();s["action_mask"].zero_()
    with pytest.raises(ValueError,match="mask"):m(**s)
