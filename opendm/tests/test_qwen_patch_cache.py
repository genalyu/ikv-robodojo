"""Regression: patch branches must not mutate shared Qwen hybrid prefix state."""
from types import SimpleNamespace
import torch
from transformers.cache_utils import DynamicLayer,LinearAttentionLayer
from opendm.model.dm05.qwen_patch_scorer import branch_cache

def test_branch_cache_isolation():
    attention=DynamicLayer()
    attention.update(torch.randn(1,2,3,4),torch.randn(1,2,3,4))
    linear=LinearAttentionLayer()
    linear.update_conv_state(torch.randn(1,8,4))
    linear.update_recurrent_state(torch.randn(1,2,4,4))
    original=SimpleNamespace(layers=[attention,linear])
    saved=(attention.keys.clone(),attention.values.clone(),
           linear.conv_states[0].clone(),linear.recurrent_states[0].clone())
    for _ in range(3):
        child=branch_cache(original)
        child.layers[0].update(torch.randn(1,2,1,4),torch.randn(1,2,1,4))
        child.layers[1].update_conv_state(torch.randn(1,8,2))
        child.layers[1].update_recurrent_state(torch.randn(1,2,4,4))
        assert child.layers[0].keys.shape[-2]==4
        for actual,expected in zip((attention.keys,attention.values,linear.conv_states[0],linear.recurrent_states[0]),saved):
            assert torch.equal(actual,expected),"branch mutated the shared native prefix"
    print("PASS shared attention KV and linear recurrent/conv state remain unchanged")

if __name__=="__main__":test_branch_cache_isolation()
