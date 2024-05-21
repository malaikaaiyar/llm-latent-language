import torch
import einops
from torch import FloatTensor, Tensor
from tuned_lens import TunedLens
from typing import Tuple, List, Optional, Dict, Callable, Iterable
from jaxtyping import Int, Float
# %%
import torch.nn as nn
from transformer_lens import HookedTransformer
MAIN = __name__ == "__main__"
# %%

class BatchTunedLens(nn.Module):
    """
    A module that applies a batch of tuned lenses to the input tensor.

    Args:
        tuned_lens (TunedLens): The tuned lens object containing the layer translators.

    Attributes:
        unembed (callable): The unembedding function.

        W_lens (nn.Parameter): The weight parameters for the tuned lenses.
        b_lens (nn.Parameter): The bias parameters for the tuned lenses.
    """

    def __init__(self, tuned_lens):
        super(BatchTunedLens, self).__init__()
        self.unembed = tuned_lens.unembed
        
        num_lenses = len(tuned_lens.layer_translators)
        out_features, in_features = tuned_lens.layer_translators[0].weight.shape
        
        device = tuned_lens.layer_translators[0].weight.device  # Extract device from layer_lens.weight
        dtype = tuned_lens.layer_translators[0].weight.dtype    # Extract dtype from layer_lens.weight

        self.W_lens = nn.Parameter(torch.empty((num_lenses, out_features, in_features), device=device, dtype=dtype))
        self.b_lens = nn.Parameter(torch.empty((num_lenses, out_features), device=device, dtype=dtype))
        
        for i in range(num_lenses):
            self.W_lens[i].data.copy_(tuned_lens.layer_translators[i].weight)
            self.b_lens[i].data.copy_(tuned_lens.layer_translators[i].bias)
        
    def forward(self, h : Float[Tensor, "... num_layers dmodel"], skip_unembed = False):
        """
        Forward pass of the BatchTunedLens module.

        Args:
            h (torch.Tensor): The input tensor of shape `... num_layers dmodel`.
            skip_unembed (bool, optional): Whether to skip the unembedding step. Defaults to False.

        Returns:
            torch.Tensor: The output tensor after applying the tuned lenses.
        """
        # Note that we add the translator output residually, in contrast to the formula
        # in the paper. By parametrizing it this way we ensure that weight decay
        # regularizes the transform toward the identity, not the zero transformation.
        # See https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/nn/lenses.py#L311C32-L312C1
        # nn.Linear uses fused-multiply torch.addmm
        h_out = einops.einsum(h, self.W_lens, "... layers din, layers dout din -> ... layers dout") + self.b_lens
        new_h = h + h_out
        if skip_unembed:
            return new_h
        else:
            return self.unembed(new_h)


def load_tuned_lens(model):
    """
    Load a tuned lens model.

    Args:
        model_name (str): The name or path of the model.
        zero_last_lens (bool, optional): Whether to zero the last lens layer. Defaults to True.

    Returns:
        TunedLens: The loaded tuned lens model.
    """
    device = next(model.parameters()).device
    if isinstance(model, HookedTransformer):
        model.config = model.cfg 
        model.config.name_or_path = model.cfg.tokenizer_name

    tuned_lens = TunedLens.from_model_and_pretrained(model)
    tuned_lens.to(device)
    return tuned_lens        

if MAIN:
    device = "cuda:0"
    tuned_lens = load_tuned_lens("meta-llama/Llama-2-7b-hf")
    tuned_lens.to(device)
    batched_tuned_lens = BatchTunedLens(tuned_lens).to(device)

    for i in range(32):
        assert torch.equal(batched_tuned_lens.W_lens[i].data, tuned_lens.layer_translators[i].weight.data), "weight no match"
        assert torch.equal(batched_tuned_lens.b_lens[i].data, tuned_lens.layer_translators[i].bias.data), "bias no match"
    print("weight/bias match!")

    resid = torch.randn(32, 4096, dtype = torch.float16, device=device)

    for i in range(32):
        #y = ((resid @ batched_tuned_lens.W_lens[i].T) + batched_tuned_lens.b_lens[i])
        #y2 = resid @ tuned_lens.layer_translators[0].weight.T + tuned_lens.layer_translators[0].bias
        y2 = resid[i] + tuned_lens.layer_translators[i](resid[i])
        y3 = batched_tuned_lens(resid, skip_unembed=True)[i]
        assert torch.allclose(y2,y3, rtol=0.05, atol = 1e-4), f"failure in layer {i} {y2=} {y3=}"
    print("linear match!")
    
    h = torch.randn(32, 4096).half().to(device)
    y = batched_tuned_lens(h)
    y2 = torch.stack([tuned_lens(h[i],i) for i in range(32)], dim=0)
    print(f"{y.shape=}, {y2.shape=}")
    torch.allclose(y,y2, rtol=0.05, atol = 1e-4)
            
