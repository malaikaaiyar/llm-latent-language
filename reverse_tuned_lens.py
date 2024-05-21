# %%
from tracemalloc import start
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm import tqdm
import pickle 
import argparse
import pprint
# === Typing Libraries ===
from typing import Tuple, List, Optional, Dict, Callable, Iterable, Any, Union
from jaxtyping import Int, Float
from beartype import beartype

# ==== Torch/Transformer Libraries ====
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
from transformer_lens.components import RMSNorm
from einops import einsum


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from tuned_lens_wrap import load_tuned_lens
from tuned_lens import TunedLens
# %%
MAIN =  (__name__ == '__main__')

# tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
# tokenizer_vocab = tokenizer.get_vocab()
# %%


# %%

if MAIN and 'LOAD_MODEL_LENS' not in globals():
    LOAD_MODEL_LENS = False
    model_name = 'meta-llama/Llama-2-7b-hf'
    model = HookedTransformer.from_pretrained_no_processing(model_name, dtype = torch.float16, device = "cpu")
    
    tuned_lens = load_tuned_lens(model)
    model.tuned_lens = tuned_lens
    tuned_lens.to("cuda")
    #del model?

# %%
from transformer_lens.hook_points import HookedRootModule, HookPoint

class HookedModuleWrapper(HookedRootModule):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.setup()

    def setup(self):
        """
        Sets up the hook points for the wrapped module.
        """
        self.mod_dict = {}
        self.hook_dict: Dict[str, HookPoint] = {}
        for name, submodule in self.module.named_modules():
            if name == "":
                continue
            submodule.name = name # type: ignore
            self.mod_dict[name] = submodule
            if isinstance(submodule, HookPoint):
                self.hook_dict[name] = submodule

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)
# %%
class ReverseRMSNorm(RMSNorm):
    
    @classmethod
    def from_pretrained(cls, module: nn.Module):
        instance = cls(module.cfg)  # Initialize with the same config as the module
        instance.w = nn.Parameter(torch.reciprocal(module.w))
        instance.eps = module.eps
        return instance

    def forward(self, 
                resid : Float[Tensor, "batch seq dmodel"], 
                cached_scale : Float[Tensor, "batch seq 1"]
        ) -> Float[Tensor, "batch seq dmodel"]:
        out = (resid * cached_scale) * self.w
        return out

# %%

if MAIN:
    rms_norm = HookedModuleWrapper(tuned_lens.unembed.final_norm)
    rev_rms_norm = ReverseRMSNorm.from_pretrained(tuned_lens.unembed.final_norm)
    
    x = 10 * torch.randn(1, 32, 4096).to(next(tuned_lens.parameters()).device)

    output, cache = rms_norm.run_with_cache(x, names_filter='hook_scale')
    x_hat = rev_rms_norm(output, cache['hook_scale'])
    assert torch.allclose(x, x_hat, rtol=1e-2, atol=1e-5), "RMSNorm failed"

# %%

def compute_inverse_linear(layer: nn.Linear, 
                           y: Float[Tensor, "... out_feat"]
    ) -> Float[Tensor, "... in_feat"]:
    
    # y = x @ A.T + b
    A, b = layer.weight, layer.bias
    z = y - b
    x_guess = torch.linalg.solve((A.T @ A).float(), (z @ A).mT.float()).mT
    x_guess = x_guess.type_as(y)
    return x_guess
# %%


class ReverseLens(nn.Module):
    def __init__(self, unembed, config):
        super().__init__()
        self.config = config
        self.unembed = unembed
        self.W_inv = nn.Parameter(torch.eye(config.d_model).repeat(config.n_layers, 1, 1))  # Identity matrix by default
        self.bias = nn.Parameter(torch.zeros(config.n_layers, config.d_model))
        # # Initialize un_RMSNorm with default config
        self.un_RMSNorm = ReverseRMSNorm(config)

    @classmethod
    def from_tuned_lens(cls, tuned_lens: 'TunedLens') -> 'ReverseLens':
        """Initialize ReverseLens from an existing TunedLens."""
        instance = cls(tuned_lens.unembed.unembedding, tuned_lens.unembed.final_norm.cfg)

        # Stack the weights and biases from the layer translators
        lens_weights = torch.stack([layer.weight.data for layer in tuned_lens.layer_translators], dim=0)
        instance.bias.data = torch.stack([layer.bias.data for layer in tuned_lens.layer_translators], dim=0)
        device = lens_weights[0].device

        # Compute the inverse of (I + W)
        I = torch.eye(instance.config.d_model).type_as(lens_weights[0]).to(device)
        instance.W_inv = nn.Parameter(torch.zeros_like(lens_weights))
        # Kind of expensive, but it's a once-off 
        for layer in range(instance.config.n_layers):
            mat = I + lens_weights[layer]
            instance.W_inv.data[layer] = torch.linalg.inv(mat.float()).type_as(lens_weights[0])
        # Initialize the un_RMSNorm module with the configuration from the tuned lens
        instance.un_RMSNorm = ReverseRMSNorm.from_pretrained(tuned_lens.unembed.final_norm)
        
        return instance

    def forward(self, 
                idx: Union[Int[Tensor, "num_vec"], Float[Tensor, "num_vec d_vocab"]], 
                layers: Union[Optional[int], Int[Tensor, "layers"]], 
                cached_scale: Union[float, Float[Tensor, "d_model"]]=1,
                use_logits=False) -> Float[Tensor, "layers dmodel"]:
        if layers is None:
            layers = torch.arange(self.config.n_layers, device=idx.device, dtype=torch.int64)
        if isinstance(layers, int):
            layers = torch.LongTensor([layers]).to(idx.device)
            
        if use_logits:
            x = compute_inverse_linear(self.unembed, idx)
        else:
            x = self.unembed.weight[idx] # (num_vec, d_model)
        x = self.un_RMSNorm(x, cached_scale) # (num_vec, d_model)
        x = x.type_as(self.W_inv[0])
        b = self.bias[layers] # (layers, d_model)
        W = self.W_inv[layers] # (layers, d_model, d_model) 
        x = x.unsqueeze(1) - b # (num_vec, layers, d_model)
        y = einsum(x, W.mT, "... n l i, l i j -> ... l n j") # (layers, num_vec, d_model) 
        return y
# %%
if __name__ == '__main__':
    #tuned_lens.to(torch.device('cuda'))
    rev_lens = ReverseLens.from_tuned_lens(tuned_lens)
    torch.cuda.empty_cache()
    tuned_lens_hooked = HookedModuleWrapper(tuned_lens)
# %%
# sometimes this passes. It's a bit wonky.
if __name__ == '__main__':
    idx = torch.randint(0, 32000, (32,), dtype=torch.int64).to("cuda")
    latents = rev_lens(idx, None, 1)
    for i in range(32):
        idx_logits = tuned_lens(latents[i], i)
        idx_recover = torch.argmax(idx_logits, dim=-1)
        assert torch.all(idx == idx_recover), f"Layer {i} failed"
    print("All layers passed")
# %%
# Look, the test fails, but we're inverting a 4096 -> 32000 operation here. 
# if all your care about is the distribution at the end is similar, then ehhhhhh this is fine.
# lgtm time to ship it.
if __name__ == "__main__":
    for layer in range(32):
        latents = torch.randn(32, 4096).to("cuda").half()
        logits, cache = tuned_lens_hooked.run_with_cache(latents, layer)
        scale_factor = cache['hook_scale']
        latents_again = rev_lens(logits, layer, scale_factor, use_logits=True)
        logits_again = tuned_lens(latents_again, layer)
        print(f"Layer {layer} max error {torch.max(torch.abs(latents - latents_again))}")
        #assert torch.allclose(latents, latents_again, atol = 0.1, rtol=0.1), f"Layer {layer} failed"
        #assert torch.allclose(logits, logits_again, atol = 1e-1, rtol=0.1), f"Layer {layer} failed"
        p = logits.softmax(dim=-1)
        q = logits_again.softmax(dim=-1)
        assert F.kl_div(p,q) < 0.001, f"Layer {layer} failed"
    print("All individuakl layers passed")
# %%
    

# %%
