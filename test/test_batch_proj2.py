
# %%
import pytest
import torch
import einops
from pathlib import Path
import sys
from einops import rearrange
from torch import Tensor
from jaxtyping import Float

src_path = Path(__file__).resolve().parent.parent / 'src'
sys.path.insert(0, str(src_path))

# Import the original proj function

# ===================================

def ensure_3d(x):
    dims_to_add = max(0, 3 - x.dim())
    new_shape = (1,) * dims_to_add + x.shape
    return x.view(*new_shape)

def proj(x : Float[Tensor, "batch dmodel"], 
         Y : Float[Tensor, "numvec dmodel"],
         tol : float = 1e-8 
) -> Float[Tensor, "... dmodel"]:
    # Computes the projection of x onto the subspace spanned by the columns of Y
    Y = ensure_3d(Y)
        
    nan_rows = torch.all(torch.isnan(Y), dim=-1)
    Y = Y[~nan_rows]
        
    Y = Y.mT #(dmodel, numvec) #require column vectors
    
    # Solve the linear system (Y^T @ Y) @ c = Y^T @ x
    # c is the coefficients of the projection of x onto the subspace spanned by the columns of Y
    # so the projection of x onto the subspace spanned by the columns of Y is Y @ c
    if x.ndim == 1:
        x = x.unsqueeze(0)
    
    c = torch.linalg.solve(Y.mT  @ Y, (x @ Y).mT)    
    proj_x = (Y @ c).mT 
    return proj_x.squeeze()

def batched_proj(x: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if Y.ndim == 2:
        Y = Y.unsqueeze(0)
    
    batch_size = x.shape[0]
    results = []
    
    for i in range(batch_size):
        result = proj(x[i], Y[i])
        results.append(result)
    
    return torch.stack(results).squeeze()


# %%
