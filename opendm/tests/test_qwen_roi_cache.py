import torch
from transformers import Qwen3_5Config,Qwen3_5TextConfig,Qwen3_5VisionConfig,Qwen3_5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPooling
from opendm.model.dm05.qwen_patch_roi import QwenSys2,branch_cache

def test_chunked_native_roi_prefix_and_branch_match_full_prefill():
    torch.manual_seed(23)
    text=Qwen3_5TextConfig(vocab_size=64,hidden_size=32,intermediate_size=64,num_hidden_layers=2,num_attention_heads=4,num_key_value_heads=2,head_dim=8,layer_types=['linear_attention','full_attention'],linear_num_key_heads=2,linear_num_value_heads=2,linear_key_head_dim=8,linear_value_head_dim=8,linear_conv_kernel_dim=4,rope_parameters=dict(rope_type='default',rope_theta=10000.0,partial_rotary_factor=1.0,mrope_section=[1,1,2]))
    vision=Qwen3_5VisionConfig(depth=1,hidden_size=32,intermediate_size=64,num_heads=4,out_hidden_size=32,patch_size=2,spatial_merge_size=2)
    config=Qwen3_5Config(text_config=text.to_dict(),vision_config=vision.to_dict(),image_token_id=5,video_token_id=6,vision_start_token_id=8,vision_end_token_id=9)
    model=Qwen3_5ForConditionalGeneration(config).eval()
    e=QwenSys2.__new__(QwenSys2);e.model=model;e.prefill_chunk_size=3
    prefix=torch.tensor([[1,8,5,5,5,5,9,2,3]])
    suffix=torch.tensor([[4,8,5,5,5,5,9,3,7]])
    ids=torch.cat((prefix,suffix),dim=1);grid=torch.tensor([[1,4,4]])
    features=[torch.randn(4,32),torch.randn(4,32)]
    with torch.inference_mode():
     model.model.rope_deltas=None
     positions=model._prepare_position_ids_for_generation(ids,dict(attention_mask=torch.ones_like(ids),mm_token_type_ids=(ids==5).long(),image_grid_thw=torch.cat([grid,grid])))
     expected=model(input_ids=ids,position_ids=positions,attention_mask=torch.ones_like(ids),mm_encoder_outputs=dict(image=BaseModelOutputWithPooling(pooler_output=tuple(features))),logits_to_keep=1).logits
     cache=e.prefill_prefix(prefix,features[:1],grid)
     original=[{k:v.clone() for k,v in vars(layer).items() if isinstance(v,torch.Tensor)} for layer in cache.layers]
     actual=model(input_ids=suffix,position_ids=positions[...,prefix.shape[1]:],attention_mask=torch.ones_like(ids),past_key_values=branch_cache(cache),mm_encoder_outputs=dict(image=BaseModelOutputWithPooling(pooler_output=(features[1],))),logits_to_keep=1).logits
     print('MAX_LOGIT_ERROR',float((expected-actual).abs().max()))
     torch.testing.assert_close(actual,expected,rtol=1e-4,atol=1e-5)
     for layer,state in zip(cache.layers,original):
      for key,value in state.items():assert torch.equal(getattr(layer,key),value)
    print('CHUNK_AND_BRANCH_PARITY_PASSED')
    
