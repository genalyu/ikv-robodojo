"""If selection matches the baseline window, IKV must preserve policy outputs."""
import torch
import pytest
from transformers import Gemma3Config,Gemma3TextConfig,SiglipVisionConfig
from opendm.model.dm05.dm05_arch import DM05Config,DM05ForConditionalGeneration
from opendm.model.dm05.online_history import pack_history_features
from opendm.model.dm05.qwen_archive import QwenArchiveBank

@pytest.mark.parametrize("device,dtype", [("cpu",torch.float32),("cuda",torch.bfloat16)])
def test_qwen_bank_baseline_window_action_parity(device,dtype):
    if device=="cuda" and not torch.cuda.is_available():pytest.skip("CUDA is unavailable")
    torch.manual_seed(932)
    text=Gemma3TextConfig(vocab_size=128,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,layer_types=['sliding_attention','full_attention'],sliding_window=16,max_position_embeddings=1024)
    vision=SiglipVisionConfig(hidden_size=32,intermediate_size=64,num_hidden_layers=1,num_attention_heads=4,image_size=16,patch_size=4)
    cfg=DM05Config(vlm_config=Gemma3Config(text_config=text.to_dict(),vision_config=vision.to_dict(),mm_tokens_per_image=16,image_token_index=127),action_config=text,action_dim=4,chunk_size=2,ikv_rgb_enabled=True)
    model=DM05ForConditionalGeneration(cfg).to(device=device,dtype=dtype).eval()
    class BaselineWindowScorer:
        def score(self,candidates,current_time):
            return {c['id']:100*c['time']/max(current_time,1) for c in candidates}
    bank=QwenArchiveBank(320,BaselineWindowScorer(),'parity',protect_current=False)
    base=torch.tensor([[2]+[7]*320+[3]],device=device)
    for start,end in [(0,0),(0,32),(0,320),(16,336)]:
        features=torch.randn(end-start,32,device=device,dtype=dtype)
        ids,types,hm,packed=pack_history_features(base,torch.zeros_like(base),[features])
        patches=torch.arange(start,end)
        kw=dict(input_ids=ids,attention_mask=torch.ones_like(ids),token_type_ids=types,history_mask=hm,history_features=packed,diffusion_steps=2,action_mask=torch.ones(1,1,4,device=device,dtype=dtype))
        torch.manual_seed(123);expected=model.inference_action(**kw)
        req=dict(bank=bank,ids=patches,descriptors=torch.zeros(len(patches),1,dtype=torch.long),times=patches//16)
        torch.manual_seed(123);actual=model.inference_action(**kw,ikv_kv_request=req)
        print('window',start,end,'max_action_error',float((expected-actual).abs().max()),flush=True)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        assert bank.metadata['ids'].tolist()==patches.tolist()
