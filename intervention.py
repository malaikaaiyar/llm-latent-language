from typing import Any, List, Callable, Dict
from torch import Tensor
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookPoint
from dq_utils import proj
from beartype import beartype

class Intervention:
    def __init__(self, func: Callable[..., Tensor], layers: List[int]):
        self.func = func
        self.layers = layers
        
    def apply(self, resid: Tensor, hook: Any, model, **kwargs) -> Tensor:
        return self.func(resid, hook, model, **kwargs)

    def fwd_hooks(self, model: Any, **kwargs) -> List:
        temp_hook_fn = lambda resid, hook: self.apply(resid, hook, model, **kwargs)
        return [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in self.layers]

def hook_reject_subspace(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids : Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids : Int[Tensor, "num_alt_latent_tokens"] = None,
    intervention_correct_latent_space : bool = True,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    # modify attn_pattern (can be inplace)
    if intervention_correct_latent_space:
        subspace = model.unembed.W_U.T[latent_ids]
    else:
        subspace = model.unembed.W_U.T[alt_latent_ids]
        
    last_tblock = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    last_tblock = last_tblock - proj(last_tblock.float(), subspace.float())
    resid[:, -1] = last_tblock
    return resid




def hook_move_subspace(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    subspace = model.unembed.W_U.T[latent_ids]
    subspace_alt = model.unembed.W_U.T[alt_latent_ids]
    
    v = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    proj_correct = proj(v.float(), subspace.float()).half()
    #resid_alt = proj(last_tblock.float(), subspace_alt.float())
    proj_counter = proj(v.float(), subspace_alt.float()).half()
    resid[:, -1] = v - proj_correct + proj_counter
    return resid

def hook_move_subspace2(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    scale_coeff : Float = 1,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    subspace = model.unembed.W_U.T[latent_ids]
    subspace_alt = model.unembed.W_U.T[alt_latent_ids]
    
    v = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    proj_correct = proj(v.float(), subspace.float()).half()
    #resid_alt = proj(last_tblock.float(), subspace_alt.float())
    proj_counter = proj(proj_correct.float(), subspace_alt.float()).half()
    resid[:, -1] = v - proj_correct + scale_coeff * proj_counter
    return resid

def hook_reject_tok(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids : Int[Tensor, "num_latent_tokens"] = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    # modify attn_pattern (can be inplace)
    subspace = model.unembed.W_U.T[latent_ids]
    last_tblock = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    last_tblock = last_tblock - proj(last_tblock.float(), subspace.float())
    resid[:, -1] = last_tblock
    return resid