# %%
import torch
from jaxtyping import Float
from torch import Tensor
import time
from tqdm import tqdm

# David Quarel
# Testing various implementations of vector projection onto a subspace.

# proj2 is probably ideal

# All implementations produce correct results.
# proj1 execution time: 2.637066 seconds
# proj2 execution time: 0.087193 seconds
# proj3 execution time: 0.371738 seconds
# proj4 execution time: 0.412574 seconds


def proj3(x: Float[Tensor, "dmodel"], Y: Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "dmodel"]:
    Y = Y.float().T
    x = x.float()

    # Solve the least squares problem Y @ c = x
    
    
    
    c = torch.linalg.lstsq(Y, x.T).solution

    # Compute the projection of x onto Span(Y)
    proj_x = Y @ c
    return proj_x.T.squeeze()


# %%
def proj2(x : Float[Tensor, "batch dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "dmodel"]:
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
    c = torch.linalg.solve(Y.T @ Y, (x @ Y).T).squeeze()
    return (Y @ c).T

# %%
def proj1(x : Float[Tensor, "dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "dmodel"]:
    # Removes the projection of x onto the subspace spanned by the columns of Y
    Y = Y.float().T
    x = x.float()
    P = Y @ torch.pinverse(Y)
    return x @ P.T
# %%

def proj4(x: Float[Tensor, "batch dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "batch dmodel"]:
    # Computes the projection of x onto the subspace spanned by the columns of Y
    Y = Y.float().T  # (numvec, dmodel)
    x = x.float()  # (..., dmodel)
    
    # Compute the projection using torch.linalg.lstsq
    proj_coeffs = torch.linalg.lstsq(Y, x.T).solution  # (..., numvec)
    proj_x = proj_coeffs.T @ Y.T  # (..., dmodel)
    
    return proj_x

# %%

# Test parameters
numvec = 200
dmodel = 40960
batch=100

# Generate random input data
x = torch.randn(batch,dmodel)
Y = torch.randn(numvec, dmodel)

# Test equivalence
proj_x1 = proj1(x, Y)
proj_x2 = proj2(x, Y)
proj_x3 = proj3(x, Y)
proj_x4 = proj4(x, Y)

tolerance = 1e-5  # Adjust the tolerance as needed

# Print shapes of each
print("Shape of proj_x1:", proj_x1.shape)
print("Shape of proj_x2:", proj_x2.shape)
print("Shape of proj_x3:", proj_x3.shape)
print("Shape of proj_x4:", proj_x4.shape)


# Test proj2 against proj1
proj_x2_test = proj2(x, Y)
assert torch.allclose(proj_x2_test, proj_x1, rtol=tolerance, atol=tolerance), "proj2 output is not close enough to proj1"

# Test proj3 against proj1
proj_x3_test = proj3(x, Y)
assert torch.allclose(proj_x3_test, proj_x1, rtol=tolerance, atol=tolerance), "proj3 output is not close enough to proj1"

# Test proj4 against proj1
proj_x4_test = proj4(x, Y)
assert torch.allclose(proj_x4_test, proj_x1, rtol=tolerance, atol=tolerance), "proj4 output is not close enough to proj1"


print("All implementations produce identical results.")
# %%
# Profile the code
from tqdm import tqdm
import torch
import time
from tqdm import tqdm
import random
num_runs = 1
# Test parameters


# Generate random input data
x = torch.randn(batch, dmodel)
Y = torch.randn(numvec, dmodel)

# Test proj1
proj_x1 = proj1(x, Y)
assert proj_x1.shape == x.shape, "proj1 output has incorrect shape"

# Test proj2
proj_x2 = proj2(x, Y)
assert proj_x2.shape == x.shape, "proj2 output has incorrect shape"

# Test proj3
proj_x3 = proj3(x, Y)
assert proj_x3.shape == x.shape, "proj3 output has incorrect shape"

# Test proj4
proj_x4 = proj4(x, Y)
assert proj_x4.shape == x.shape, "proj4 output has incorrect shape"

# Define tolerance for floating-point comparisons
tolerance = 1e-5

# Test proj2 against proj1
proj_x2_test = proj2(x, Y)
assert torch.allclose(proj_x2_test, proj_x1, rtol=tolerance, atol=tolerance), "proj2 output is not close enough to proj1"

# Test proj3 against proj1
proj_x3_test = proj3(x, Y)
assert torch.allclose(proj_x3_test, proj_x1, rtol=tolerance, atol=tolerance), "proj3 output is not close enough to proj1"

# Test proj4 against proj1
proj_x4_test = proj4(x, Y)
assert torch.allclose(proj_x4_test, proj_x1, rtol=tolerance, atol=tolerance), "proj4 output is not close enough to proj1"

print("All implementations produce correct results.")

# Profile the code
num_runs = 1

functions = [proj1, proj2, proj3, proj4]

for func in functions:
    start_time = time.time()
    for i in range(num_runs):
        torch.manual_seed(i)  # Set seed for reproducibility
        x = torch.randn(batch, dmodel)
        Y = torch.randn(numvec, dmodel)
        func(x, Y)
    end_time = time.time()
    print(f"{func.__name__} execution time: {(end_time - start_time):.6f} seconds")
# %%