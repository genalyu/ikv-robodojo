import jax,numpy as np
from openpi.models.pi0_config import Pi0Config
from openpi.shared import nnx_utils

def test_external_prefill_preserves_first_frame_denoising():
    with jax.default_device(jax.devices("cpu")[0]):
        cfg=Pi0Config(pi05=True,paligemma_variant='dummy',action_expert_variant='dummy',action_dim=8,action_horizon=2,max_token_len=8,ikv_rgb_enabled=True,ikv_single_frame=True,ikv_history_capacity=768)
        m=cfg.create(jax.random.key(0));obs=cfg.fake_obs(1)
        f=nnx_utils.module_jit(m.sample_actions_single_frame_ikv);p=nnx_utils.module_jit(m.qwen_prefill)
        a,_=f(jax.random.key(1),obs,None,num_steps=2)
        cache=p(obs)
        b,_=f(jax.random.key(1),obs,None,num_steps=2,external_cache=cache)
        np.testing.assert_allclose(np.asarray(a),np.asarray(b),rtol=1e-5,atol=1e-5)
        print('PI05 external-cache denoising first-frame parity PASS')
