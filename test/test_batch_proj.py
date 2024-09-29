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

# ===========================================

def preprocess_pad_subspaces(V_list):
    """
    Pad a list of subspace matrices into a single 3D tensor, using NaNs for padding.
    
    Args:
    V_list (list of torch.Tensor): List of matrices, each of shape (num_vec, dmodel)
    
    Returns:
    torch.Tensor: Padded tensor of shape (batch, max_num_vec, dmodel)
    """
    batch = len(V_list)
    dmodel = V_list[0].shape[1]
    max_num_vec = max(v.shape[0] for v in V_list)
    
    # Initialize the padded tensor with NaNs
    V_padded = torch.full((batch, max_num_vec, dmodel), float('nan'), device=V_list[0].device)
    
    # Fill in the actual vectors
    for i, v in enumerate(V_list):
        V_padded[i, :v.shape[0], :] = v
    
    return V_padded

def preprocess_orthogonalize_subspaces(V, eps=1e-8):
    """
    Orthogonalize NaN-padded subspaces using SVD.
    
    Args:
    V (torch.Tensor): Tensor of shape (batch, num_vec, dmodel) with NaN padding
    eps (float): Threshold for singular values, used for numerical stability and rank determination
    
    Returns:
    torch.Tensor: Orthogonalized subspaces, shape (batch, num_vec, dmodel)
    torch.Tensor: Mask indicating valid vectors, shape (batch, num_vec)
    """
    
    
    batch, num_vec, dmodel = V.shape
    
    # Create mask for non-NaN vectors
    mask = ~torch.isnan(V).any(dim=-1)  # shape: (batch, num_vec)
    
    # Replace NaNs with zeros for computation
    V_clean = torch.where(torch.isnan(V), torch.zeros_like(V), V)
    
    # Normalize non-zero vectors
    norms = torch.norm(V_clean, dim=2, keepdim=True)
    V_normalized = torch.where(norms > eps, V_clean / norms, V_clean)
    
    # Compute batched SVD
    U, S, _ = torch.linalg.svd(V_normalized, full_matrices=False)
    
    # Create mask for significant singular values
    s_mask = S > eps
    
    # Apply both masks
    Q = U * s_mask.unsqueeze(-1)
    mask = mask & s_mask
    
    # Ensure NaN padding is preserved
    Q = torch.where(mask.unsqueeze(-1), Q, torch.full_like(Q, float('nan')))
    
    return Q, mask

def nan_safe_gpu_subspace_projection(x, Q, mask):
    """
    NaN-safe GPU-optimized projection of each vector in x onto its corresponding preprocessed subspace.
    
    Args:
    x (torch.Tensor): Batch of vectors to project, shape (batch, dmodel)
    Q (torch.Tensor): Preprocessed (orthogonalized) subspaces, shape (batch, num_vec, dmodel)
    mask (torch.Tensor): Mask indicating valid vectors, shape (batch, num_vec)
    
    Returns:
    torch.Tensor: Batch of projected vectors, shape (batch, dmodel)
    """
    # Compute dot products between x and each basis vector in Q
    dots = torch.bmm(x.unsqueeze(1), Q.transpose(1, 2))  # (batch, 1, num_vec)
    
    # Apply mask to zero out invalid dot products (including NaNs)
    dots = torch.where(mask.unsqueeze(1), dots, torch.zeros_like(dots))
    
    # Compute the projection by multiplying dots with Q
    projections = torch.bmm(dots, torch.where(torch.isnan(Q), torch.zeros_like(Q), Q))  # (batch, 1, dmodel)
    
    return projections.squeeze(1)  # (batch, dmodel)

def proj2(x, V):
    Q, mask =preprocess_orthogonalize_subspaces(V)
    return nan_safe_gpu_subspace_projection(x, Q, mask)
    


# Helper function to ensure tensors are close
def assert_close(a, b, rtol=1e-4, atol=1e-7):
    assert torch.allclose(a, b, rtol=rtol, atol=atol), f"Tensors not close:\n{a}\n{b}"

# Create a batched version of proj

@pytest.fixture
def rng():
    return torch.manual_seed(42)

def test_single_vector(rng):
    x = torch.randn(5)
    Y = torch.randn(1, 5)
    
    result1 = batched_proj(x, Y)
    result2 = proj2(x, Y)
    
    assert_close(result1, result2)

def test_multiple_vectors(rng):
    x = torch.randn(10)
    Y = torch.randn(3, 10)
    
    result1 = batched_proj(x, Y)
    result2 = proj2(x, Y)
    
    assert_close(result1, result2)

def test_batched_input(rng):
    x = torch.randn(4, 7)
    Y = torch.randn(4, 2, 7)
    
    result1 = batched_proj(x, Y)
    result2 = proj2(x, Y)
    
    assert_close(result1, result2)

def test_different_vector_counts(rng):
    x = torch.randn(3, 8)
    Y = torch.full((3, 5, 8), float('nan'))
    Y[0, :2] = torch.randn(2, 8)
    Y[1, :3] = torch.randn(3, 8)
    Y[2, :4] = torch.randn(4, 8)
    
    result1 = batched_proj(x, Y)
    result2 = proj2(x, Y)
    
    assert_close(result1, result2)

def test_edge_case_single_dimension(rng):
    x = torch.randn(1)
    Y = torch.randn(1, 1)
    
    result1 = batched_proj(x, Y)
    result2 = proj2(x, Y)
    
    assert_close(result1, result2)

def test_edge_case_zero_vectors(rng):
    x = torch.randn(6)
    Y = torch.full((1, 0, 6), float('nan'))
    
    result1 = batched_proj(x, Y)
    result2 = proj2(x, Y)
    
    assert_close(result1, result2)
    assert_close(result1, torch.zeros_like(x))

def test_large_batch_different_vector_counts(rng):
    batch_size = 10
    max_vectors = 8
    dim = 12
    
    x = torch.randn(batch_size, dim)
    Y = torch.full((batch_size, max_vectors, dim), float('nan'))
    
    for i in range(batch_size):
        num_vectors = torch.randint(1, max_vectors + 1, (1,)).item()
        Y[i, :num_vectors] = torch.randn(num_vectors, dim)
    
    result1 = batched_proj(x, Y)
    result2 = proj2(x, Y)
    
    assert_close(result1, result2)

def test_numerical_stability(rng):
    x = torch.randn(5) * 1e6
    Y = torch.randn(2, 5) * 1e-6
    
    result1 = batched_proj(x, Y)
    result2 = proj2(x, Y)
    
    assert_close(result1, result2, rtol=1e-4, atol=1e-4)  # Relaxed tolerance due to potential numerical issues

if __name__ == "__main__":
    pytest.main([__file__])
# %%
