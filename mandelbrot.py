import torch
import matplotlib.pyplot as plt

def mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter):
    """
    Generate the Mandelbrot set using PyTorch tensor operations.

    Args:
    - xmin, xmax: The range on the real axis.
    - ymin, ymax: The range on the imaginary axis.
    - width, height: The dimensions of the output image.
    - max_iter: The maximum number of iterations to determine divergence.

    Returns:
    A height x width array of integers indicating the iteration count at which the corresponding point diverges.
    """
    # Create a mesh grid for the complex plane
    x = torch.linspace(xmin, xmax, width, dtype=torch.float32)
    y = torch.linspace(ymin, ymax, height, dtype=torch.float32)
    Y, X = torch.meshgrid(y, x)
    Z = X + 1j * Y

    # Initialize tensors to keep track of divergence
    C = Z.clone()
    diverged = torch.zeros(Z.shape, dtype=torch.int)
    mask = torch.ones(Z.shape, dtype=torch.bool)

    for i in range(max_iter):
        # Perform the iteration Z = Z^2 + C
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        
        # Update the mask where divergence criterion is met
        diverged_mask = (Z.abs() > 2) & mask
        diverged[diverged_mask] = i
        mask &= ~diverged_mask

    return diverged

# Parameters for the set
xmin, xmax = -2.0, 0.5
ymin, ymax = -1.25, 1.25
width, height = 1024, 1024
max_iter = 100

# Generate the set
mandelbrot = mandelbrot_set(xmin, xmax, ymin, ymax, width, height, max_iter)

# Plotting
plt.figure(figsize=(10,10))
plt.imshow(mandelbrot.numpy(), cmap='inferno', extent=(xmin, xmax, ymin, ymax))
plt.colorbar()
plt.title("Mandelbrot Set")
plt.show()
