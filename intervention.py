from typing import Any, List, Callable, Dict, Iterable
from torch import Tensor
import torch
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookPoint
from dq_utils import proj
from beartype import beartype
import inspect
# class Intervention:
#     def __init__(self, func: Callable[..., Tensor], layers: List[int]):
#         self.func = func
#         self.layers = layers
        
#     def apply(self, resid: Tensor, hook: Any, model, **kwargs) -> Tensor:
#         return self.func(resid, hook, model, **kwargs)

#     def fwd_hooks(self, model: Any, **kwargs) -> List:
#         temp_hook_fn = lambda resid, hook: self.apply(resid, hook, model, **kwargs)
#         return [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in self.layers]

class Intervention:
    def __init__(self, func_name: str, layers: Iterable[int]):
        # Check if the function name exists in the global scope and is callable
        if func_name in globals() and callable(globals()[func_name]):
            self.func = globals()[func_name]
            # Attempt to retrieve the source code of the function
            try:
                self.description = inspect.getsource(self.func)
            except OSError:
                self.description = f"Source code for '{func_name}' not available."
        else:
            raise ValueError(f"No such function '{func_name}' found in global scope.")

        self.layers = list(layers)
        self.func_name = func_name
        
    def apply(self, resid: Tensor, hook: Any, model, **kwargs) -> Tensor:
        return self.func(resid, hook, model, **kwargs)

    def fwd_hooks(self, model: Any, **kwargs) -> List:
        # Using a lambda to capture the current model and additional args
        temp_hook_fn = lambda resid, hook: self.apply(resid, hook, model, **kwargs)
        return [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in self.layers]

# Example usage assuming hook functions are defined and available globally
# hook_reject_subspace is supposed to be a previously defined function


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

def hook_diff_subspace(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    steer_scale_coeff = kwargs.get('steer_scale_coeff', None)
    assert steer_scale_coeff is not None, "steer_scale_coeff must be provided"
    subspace_latent = model.unembed.W_U.T[latent_ids]
    latent_vec = subspace_latent.mean(dim=0)
    alt_latent_vec = model.unembed.W_U.T[alt_latent_ids].mean(dim=0)
    v = resid[:, -1]
    proj_latent = proj(v.float(), subspace_latent.float()).half()
    #print(v.shape, latent_vec.shape, alt_latent_vec.shape)
    resid[:, -1] =  v - proj_latent + steer_scale_coeff * torch.linalg.norm(proj_latent) * (alt_latent_vec - latent_vec)
    return resid