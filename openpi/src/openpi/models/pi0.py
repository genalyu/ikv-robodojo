import logging

import einops
import flax.nnx as nnx
import flax.nnx.bridge as nnx_bridge
import jax
import jax.numpy as jnp
from typing_extensions import override

from openpi.models import model as _model
from openpi.models import pi0_config
import openpi.models.gemma as _gemma
import openpi.models.siglip as _siglip
from openpi.models.ikv_rgb import compact_jax_prefix_tokens, memory_keys, merge_single_frame_jax, select_jax
from openpi.shared import array_typing as at

logger = logging.getLogger("openpi")


def make_attn_mask(input_mask, mask_ar):
    """Adapted from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` bool[?B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: bool[?B, N] mask that's true where previous tokens cannot depend on
        it and false where it shares the same attention mask as the previous token.
    """
    mask_ar = jnp.broadcast_to(mask_ar, input_mask.shape)
    cumsum = jnp.cumsum(mask_ar, axis=1)
    attn_mask = cumsum[:, None, :] <= cumsum[:, :, None]
    valid_mask = input_mask[:, None, :] * input_mask[:, :, None]
    return jnp.logical_and(attn_mask, valid_mask)


@at.typecheck
def posemb_sincos(
    pos: at.Real[at.Array, " b"], embedding_dim: int, min_period: float, max_period: float
) -> at.Float[at.Array, "b {embedding_dim}"]:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if embedding_dim % 2 != 0:
        raise ValueError(f"embedding_dim ({embedding_dim}) must be divisible by 2")

    fraction = jnp.linspace(0.0, 1.0, embedding_dim // 2)
    period = min_period * (max_period / min_period) ** fraction
    sinusoid_input = jnp.einsum(
        "i,j->ij",
        pos,
        1.0 / period * 2 * jnp.pi,
        precision=jax.lax.Precision.HIGHEST,
    )
    return jnp.concatenate([jnp.sin(sinusoid_input), jnp.cos(sinusoid_input)], axis=-1)


class Pi0(_model.BaseModel):
    def __init__(self, config: pi0_config.Pi0Config, rngs: nnx.Rngs):
        super().__init__(config.action_dim, config.action_horizon, config.max_token_len)
        self.pi05 = config.pi05
        self.ikv_rgb_enabled = config.ikv_rgb_enabled
        self.ikv_single_frame = config.ikv_single_frame
        self.ikv_history_capacity = config.ikv_history_capacity
        self.ikv_top_k = config.ikv_top_k
        self.ikv_motion_threshold = config.ikv_motion_threshold
        self.ikv_motion_only = config.ikv_motion_only
        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)
        # TODO: rewrite gemma in NNX. For now, use bridge.
        llm = nnx_bridge.ToNNX(
            _gemma.Module(
                configs=[paligemma_config, action_expert_config],
                embed_dtype=config.dtype,
                adarms=config.pi05,
            )
        )
        llm.lazy_init(rngs=rngs, method="init", use_adarms=[False, True] if config.pi05 else [False, False])
        img = nnx_bridge.ToNNX(
            _siglip.Module(
                num_classes=paligemma_config.width,
                variant="So400m/14",
                pool_type="none",
                scan=True,
                dtype_mm=config.dtype,
            )
        )
        img.lazy_init(next(iter(config.fake_obs().images.values())), train=False, rngs=rngs)
        self.PaliGemma = nnx.Dict(llm=llm, img=img)
        self.action_in_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
        if config.pi05:
            self.time_mlp_in = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        else:
            self.state_proj = nnx.Linear(config.action_dim, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_in = nnx.Linear(2 * action_expert_config.width, action_expert_config.width, rngs=rngs)
            self.action_time_mlp_out = nnx.Linear(action_expert_config.width, action_expert_config.width, rngs=rngs)
        self.action_out_proj = nnx.Linear(action_expert_config.width, config.action_dim, rngs=rngs)

        # This attribute gets automatically set by model.train() and model.eval().
        self.deterministic = True

    @at.typecheck
    def embed_prefix(
        self, obs: _model.Observation
    ) -> tuple[at.Float[at.Array, "b s emb"], at.Bool[at.Array, "b s"], at.Bool[at.Array, " s"]]:
        input_mask = []
        ar_mask = []
        tokens = []
        keys = memory_keys(obs.images) if self.ikv_rgb_enabled else []
        memory_keep = None
        # embed images
        for name in obs.images:
            image_tokens, _ = self.PaliGemma.img(obs.images[name], train=False)

            if name in keys and memory_keep is None:
                history = jnp.stack([obs.images[key] for key in keys], axis=1)
                valid = jnp.stack([obs.image_masks[key] for key in keys], axis=1)
                memory_keep = select_jax(
                    history, valid, image_tokens.shape[1],
                    capacity=self.ikv_history_capacity,
                    top_k=self.ikv_top_k,
                    threshold=self.ikv_motion_threshold,
                    motion_only=self.ikv_motion_only,
                    dino=obs.ikv_dino_features,
                    reference_dino=obs.ikv_reference_dino,
                    current_rgb=obs.ikv_current_rgb,
                )

            tokens.append(image_tokens)
            patch_mask = einops.repeat(obs.image_masks[name], "b -> b s", s=image_tokens.shape[1])
            if name in keys:
                patch_mask = patch_mask & memory_keep[:, keys.index(name)]
            input_mask.append(patch_mask)
            # image tokens attend to each other
            ar_mask += [False] * image_tokens.shape[1]

        # add language (aka tokenized inputs)
        if obs.tokenized_prompt is not None:
            tokenized_inputs = self.PaliGemma.llm(obs.tokenized_prompt, method="embed")
            tokens.append(tokenized_inputs)
            input_mask.append(obs.tokenized_prompt_mask)
            # full attention between image and language inputs
            ar_mask += [False] * tokenized_inputs.shape[1]
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        if keys:
            patch_count = memory_keep.shape[-1]
            tokens, input_mask = compact_jax_prefix_tokens(
                tokens, input_mask,
                len(_model.IMAGE_KEYS) * patch_count,
                len(keys) * patch_count,
                self.ikv_history_capacity,
            )
            ar_mask = jnp.zeros(tokens.shape[1], dtype=jnp.bool_)
        return tokens, input_mask, ar_mask

    @at.typecheck
    def embed_suffix(
        self, obs: _model.Observation, noisy_actions: _model.Actions, timestep: at.Float[at.Array, " b"]
    ) -> tuple[
        at.Float[at.Array, "b s emb"],
        at.Bool[at.Array, "b s"],
        at.Bool[at.Array, " s"],
        at.Float[at.Array, "b emb"] | None,
    ]:
        input_mask = []
        ar_mask = []
        tokens = []
        if not self.pi05:
            # add a single state token
            state_token = self.state_proj(obs.state)[:, None, :]
            tokens.append(state_token)
            input_mask.append(jnp.ones((obs.state.shape[0], 1), dtype=jnp.bool_))
            # image/language inputs do not attend to state or actions
            ar_mask += [True]

        action_tokens = self.action_in_proj(noisy_actions)
        # embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = posemb_sincos(timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0)
        if self.pi05:
            # time MLP (for adaRMS)
            time_emb = self.time_mlp_in(time_emb)
            time_emb = nnx.swish(time_emb)
            time_emb = self.time_mlp_out(time_emb)
            time_emb = nnx.swish(time_emb)
            action_expert_tokens = action_tokens
            adarms_cond = time_emb
        else:
            # mix timestep + action information using an MLP (no adaRMS)
            time_tokens = einops.repeat(time_emb, "b emb -> b s emb", s=self.action_horizon)
            action_time_tokens = jnp.concatenate([action_tokens, time_tokens], axis=-1)
            action_time_tokens = self.action_time_mlp_in(action_time_tokens)
            action_time_tokens = nnx.swish(action_time_tokens)
            action_time_tokens = self.action_time_mlp_out(action_time_tokens)
            action_expert_tokens = action_time_tokens
            adarms_cond = None
        tokens.append(action_expert_tokens)
        input_mask.append(jnp.ones(action_expert_tokens.shape[:2], dtype=jnp.bool_))
        # image/language/state inputs do not attend to action tokens
        ar_mask += [True] + ([False] * (self.action_horizon - 1))
        tokens = jnp.concatenate(tokens, axis=1)
        input_mask = jnp.concatenate(input_mask, axis=1)
        ar_mask = jnp.array(ar_mask)
        return tokens, input_mask, ar_mask, adarms_cond

    @override
    def compute_loss(
        self, rng: at.KeyArrayLike, observation: _model.Observation, actions: _model.Actions, *, train: bool = False
    ) -> at.Float[at.Array, "*b ah"]:
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=train, include_memory=self.ikv_rgb_enabled
        )

        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        time_expanded = time[..., None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        # one big forward pass of prefix + suffix at once
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        input_mask = jnp.concatenate([prefix_mask, suffix_mask], axis=1)
        ar_mask = jnp.concatenate([prefix_ar_mask, suffix_ar_mask], axis=0)
        attn_mask = make_attn_mask(input_mask, ar_mask)
        positions = jnp.cumsum(input_mask, axis=1) - 1
        (prefix_out, suffix_out), _ = self.PaliGemma.llm(
            [prefix_tokens, suffix_tokens], mask=attn_mask, positions=positions, adarms_cond=[None, adarms_cond]
        )
        v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

        return jnp.mean(jnp.square(v_t - u_t), axis=-1)

    def _ikv_prefill(self, observation, ikv_state):
        """Share the exact recurrent RGB K/V selector in inference and training."""
        batch_size = observation.state.shape[0]
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        n = self.ikv_history_capacity
        if prefix_tokens.shape[1] <= n or n % len(observation.images):
            raise ValueError("IKV capacity must equal all current visual tokens and leave prompt tokens")
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, current_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        current_k, current_v = current_cache
        new_k, new_v = current_k[:, :, :n], current_v[:, :, :n]
        # Frozen SigLIP/Pali projection doubles as the index. Pool 2048 -> 64
        # deterministically to keep the recurrent selector inexpensive.
        index = prefix_tokens[:, :n].astype(jnp.float32)
        index = index.reshape(batch_size, n, 64, index.shape[-1] // 64).mean(-1)
        index = index / jnp.maximum(jnp.linalg.norm(index, axis=-1, keepdims=True), 1e-6)

        if ikv_state is None:
            generation = jnp.array(1,dtype=jnp.int32)
            chosen,selected_index,birth,latest = merge_single_frame_jax(
                None,index,jnp.array(0,dtype=jnp.int32))
            selected_k,selected_v = new_k,new_v
        else:
            old_k,old_v,old_index,old_birth,old_latest,old_generation=ikv_state
            chosen,selected_index,birth,latest = merge_single_frame_jax(
                (old_index,old_birth,old_latest),index,old_generation)
            generation=old_generation+1
            candidates_k=jnp.concatenate((old_k,new_k),axis=2)
            candidates_v=jnp.concatenate((old_v,new_v),axis=2)
            cache_idx=chosen[None,:,:,None,None]
            cache_idx=jnp.broadcast_to(cache_idx,(candidates_k.shape[0],batch_size,n,candidates_k.shape[3],candidates_k.shape[4]))
            selected_k=jnp.take_along_axis(candidates_k,cache_idx,axis=2)
            selected_v=jnp.take_along_axis(candidates_v,cache_idx,axis=2)

        text_k, text_v = current_k[:, :, n:], current_v[:, :, n:]
        action_cache = (jnp.concatenate((selected_k,text_k),axis=2),
                        jnp.concatenate((selected_v,text_v),axis=2))
        memory_mask = jnp.ones((batch_size,n),dtype=jnp.bool_)
        action_prefix_mask = jnp.concatenate((memory_mask,prefix_mask[:,n:]),axis=1)
        state=(selected_k,selected_v,selected_index,birth,latest,generation)
        return action_cache, action_prefix_mask, state

    def compute_loss_single_frame_ikv(self, rng, observation, actions, ikv_state=None):
        """Native flow matching loss with the inference selector and detached prior state."""
        if not self.ikv_single_frame:
            raise ValueError("single-frame IKV loss requires ikv_single_frame=True")
        preprocess_rng, noise_rng, time_rng = jax.random.split(rng, 3)
        observation = _model.preprocess_observation(
            preprocess_rng, observation, train=True, include_memory=False)
        batch_shape = actions.shape[:-2]
        noise = jax.random.normal(noise_rng, actions.shape)
        time = jax.random.beta(time_rng, 1.5, 1, batch_shape) * 0.999 + 0.001
        x_t = time[..., None, None] * noise + (1 - time[..., None, None]) * actions
        u_t = noise - actions
        action_cache, action_prefix_mask, new_state = self._ikv_prefill(observation, ikv_state)
        suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(observation, x_t, time)
        suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
        memory_attn = einops.repeat(action_prefix_mask,"b p -> b s p",s=suffix_tokens.shape[1])
        full_attn_mask = jnp.concatenate((memory_attn,suffix_attn_mask),axis=-1)
        suffix_positions = (jnp.sum(action_prefix_mask,axis=-1)[:,None]
                            + jnp.cumsum(suffix_mask,axis=-1)-1)
        (_,suffix_out),_ = self.PaliGemma.llm(
            [None,suffix_tokens], mask=full_attn_mask, positions=suffix_positions,
            kv_cache=action_cache, adarms_cond=[None,adarms_cond])
        v_t = self.action_out_proj(suffix_out[:,-self.action_horizon:])
        loss = jnp.mean(jnp.square(v_t-u_t),axis=-1)
        detached = jax.tree.map(jax.lax.stop_gradient,new_state)
        return loss, detached

    def sample_actions_single_frame_ikv(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        ikv_state=None,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ):
        """Recurrently retain exactly one baseline frame worth of visual K/V.

        The current three camera views are prefixed exactly once. Their deep
        PaliGemma K/V rows compete with the N live rows from the prior call;
        current text K/V remains ephemeral and is appended after the selected
        visual memory. State is detached naturally across policy calls.
        """
        if not self.ikv_single_frame:
            raise ValueError("single-frame IKV sampler requires ikv_single_frame=True")
        observation = _model.preprocess_observation(None, observation, train=False, include_memory=False)
        batch_size = observation.state.shape[0]
        if batch_size != 1:
            raise ValueError("single-frame IKV requires one recurrent stream per policy")
        if len(observation.images) != len(_model.IMAGE_KEYS):
            raise ValueError("single-frame IKV expects exactly the baseline camera views")
        dt = -1.0 / num_steps
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        action_cache, action_prefix_mask, state = self._ikv_prefill(observation, ikv_state)
        prefix_len = action_prefix_mask.shape[1]
        def step(carry):
            x_t,time=carry
            suffix_tokens,suffix_mask,suffix_ar_mask,adarms_cond=self.embed_suffix(
                observation,x_t,jnp.broadcast_to(time,batch_size))
            suffix_attn_mask=make_attn_mask(suffix_mask,suffix_ar_mask)
            memory_attn=einops.repeat(action_prefix_mask,"b p -> b s p",s=suffix_tokens.shape[1])
            full_attn_mask=jnp.concatenate((memory_attn,suffix_attn_mask),axis=-1)
            suffix_positions=jnp.sum(action_prefix_mask,axis=-1)[:,None]+jnp.cumsum(suffix_mask,axis=-1)-1
            (_,suffix_out),_=self.PaliGemma.llm([None,suffix_tokens],mask=full_attn_mask,
                positions=suffix_positions,kv_cache=action_cache,adarms_cond=[None,adarms_cond])
            v_t=self.action_out_proj(suffix_out[:,-self.action_horizon:])
            return x_t+dt*v_t,time+dt
        def cond(carry): return carry[1]>=-dt/2
        actions,_=jax.lax.while_loop(cond,step,(noise,1.0))
        return actions,state

    @override
    def sample_actions(
        self,
        rng: at.KeyArrayLike,
        observation: _model.Observation,
        *,
        num_steps: int | at.Int[at.Array, ""] = 10,
        noise: at.Float[at.Array, "b ah ad"] | None = None,
    ) -> _model.Actions:
        observation = _model.preprocess_observation(
            None, observation, train=False, include_memory=self.ikv_rgb_enabled
        )
        # note that we use the convention more common in diffusion literature, where t=1 is noise and t=0 is the target
        # distribution. yes, this is the opposite of the pi0 paper, and I'm sorry.
        dt = -1.0 / num_steps
        batch_size = observation.state.shape[0]
        if noise is None:
            noise = jax.random.normal(rng, (batch_size, self.action_horizon, self.action_dim))

        # first fill KV cache with a forward pass of the prefix
        prefix_tokens, prefix_mask, prefix_ar_mask = self.embed_prefix(observation)
        prefix_attn_mask = make_attn_mask(prefix_mask, prefix_ar_mask)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        _, kv_cache = self.PaliGemma.llm([prefix_tokens, None], mask=prefix_attn_mask, positions=positions)
        def step(carry):
            x_t, time = carry
            suffix_tokens, suffix_mask, suffix_ar_mask, adarms_cond = self.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch_size)
            )
            # `suffix_attn_mask` is shape (b, suffix_len, suffix_len) indicating how the suffix tokens can attend to each
            # other
            suffix_attn_mask = make_attn_mask(suffix_mask, suffix_ar_mask)
            # `prefix_attn_mask` is shape (b, suffix_len, prefix_len) indicating how the suffix tokens can attend to the
            # prefix tokens
            prefix_attn_mask = einops.repeat(prefix_mask, "b p -> b s p", s=suffix_tokens.shape[1])
            # `combined_mask` is shape (b, suffix_len, prefix_len + suffix_len) indicating how the suffix tokens (which
            # generate the queries) can attend to the full prefix + suffix sequence (which generates the keys and values)
            full_attn_mask = jnp.concatenate([prefix_attn_mask, suffix_attn_mask], axis=-1)
            assert full_attn_mask.shape == (
                batch_size,
                suffix_tokens.shape[1],
                prefix_tokens.shape[1] + suffix_tokens.shape[1],
            )
            # `positions` is shape (b, suffix_len) indicating the positions of the suffix tokens
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(suffix_mask, axis=-1) - 1

            (prefix_out, suffix_out), _ = self.PaliGemma.llm(
                [None, suffix_tokens],
                mask=full_attn_mask,
                positions=positions,
                kv_cache=kv_cache,
                adarms_cond=[None, adarms_cond],
            )
            assert prefix_out is None
            v_t = self.action_out_proj(suffix_out[:, -self.action_horizon :])

            return x_t + dt * v_t, time + dt

        def cond(carry):
            x_t, time = carry
            # robust to floating-point error
            return time >= -dt / 2

        x_0, _ = jax.lax.while_loop(cond, step, (noise, 1.0))
        return x_0
