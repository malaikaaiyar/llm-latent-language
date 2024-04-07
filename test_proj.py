# %%
import torch
from jaxtyping import Float
from torch import Tensor
import time
from tqdm import tqdm

from memory_profiler import profile


@profile
def proj3(x: Float[Tensor, "dmodel"], Y: Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "dmodel"]:
    Y = Y.float().T
    x = x.float()

    # Solve the least squares problem Y @ c = x
    c = torch.linalg.lstsq(Y, x.unsqueeze(1)).solution

    # Compute the projection of x onto Span(Y)
    proj_x = Y @ c
    return proj_x.squeeze()


# %%
@profile
def proj2(x : Float[Tensor, "dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "dmodel"]:
    # Computes the projection of x onto the subspace spanned by the columns of Y
    #This is slow and runs in O(d^2)? time
    # Y = Y.float().T
    # x = x.float()
    # P = Y @ torch.pinverse(Y)
    # proj_x = x - P @ x.squeeze()
    # return proj_x
    
    Y = Y.float().T #(dmodel, numvec)
    x = x.float()

    # Solve the linear system (Y^T @ Y) @ c = Y^T @ x
    # c is the coefficients of the projection of x onto the subspace spanned by the columns of Y
    # so the projection of x onto the subspace spanned by the columns of Y is Y @ c
    c = torch.linalg.solve(Y.T @ Y, Y.T @ x.unsqueeze(1)).squeeze()
    return Y @ c

# %%
@profile
def proj1(x : Float[Tensor, "dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "dmodel"]:
    # Removes the projection of x onto the subspace spanned by the columns of Y
    Y = Y.float().T
    x = x.float()
    P = Y @ torch.pinverse(Y)
    return P @ x.squeeze()

# %%

# Test parameters
numvec = 20
dmodel = 4096



# Generate random input data
x = torch.randn(dmodel)
Y = torch.randn(numvec, dmodel)

# Test equivalence
proj_x1 = proj1(x, Y)
proj_x2 = proj2(x, Y)
proj_x3 = proj3(x, Y)

tolerance = 1e-5  # Adjust the tolerance as needed

assert torch.allclose(proj_x1, proj_x2, rtol=tolerance, atol=tolerance), "proj1 and proj2 outputs are not close enough"
assert torch.allclose(proj_x1, proj_x3, rtol=tolerance, atol=tolerance), "proj1 and proj3 outputs are not close enough"
assert torch.allclose(proj_x2, proj_x3, rtol=tolerance, atol=tolerance), "proj2 and proj3 outputs are not close enough"


print("All implementations produce identical results.")
# %%
# Profile the code
from tqdm import tqdm
num_runs = 1

start_time = time.time()
for _ in tqdm(range(num_runs)):
    proj1(x, Y)
end_time = time.time()
print(f"proj1 execution time: {(end_time - start_time) / num_runs:.6f} seconds")

start_time = time.time()
for _ in tqdm(range(num_runs)):
    proj2(x, Y)
end_time = time.time()
print(f"proj2 execution time: {(end_time - start_time) / num_runs:.6f} seconds")

start_time = time.time()
for _ in tqdm(range(num_runs)):
    proj3(x, Y)
end_time = time.time()
print(f"proj3 execution time: {(end_time - start_time) / num_runs:.6f} seconds")
# %%