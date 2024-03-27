import torch
import numpy as np
import matplotlib.pyplot as plt

def initialize_grid(shape, initial_heat_source):
    """Initialize the temperature grid and apply an initial heat source."""
    grid = torch.zeros(shape)
    grid[initial_heat_source] = 1  # Apply initial heat
    return grid

def apply_mask(grid, mask):
    """Apply a mask to the grid to define the shape of the material."""
    return grid * mask

def diffuse_heat(grid, mask, diffusion_rate=0.25):
    """Apply the heat diffusion equation to update the temperature distribution."""
    # Create a padded version of the grid for boundary conditions
    padded_grid = torch.nn.functional.pad(grid, (1, 1, 1, 1), mode='constant', value=0)
    # Compute the diffusion update
    update = (padded_grid[:-2, 1:-1] + padded_grid[2:, 1:-1] + padded_grid[1:-1, :-2] + padded_grid[1:-1, 2:] - 4 * grid) * diffusion_rate
    # Update the grid, applying the mask to maintain the shape
    new_grid = grid + update
    new_grid = apply_mask(new_grid, mask)
    return new_grid
import plotly.graph_objects as go
import torch
import numpy as np

def simulate_heat_diffusion(shape, mask, initial_heat_source, steps=100, diffusion_rate=0.25):
    """Simulate heat diffusion for a given number of steps and store each step."""
    grid = initialize_grid(shape, initial_heat_source)
    grid = apply_mask(grid, mask)
    grids = [grid.numpy()]  # Store initial state
    
    for _ in range(steps):
        grid = diffuse_heat(grid, mask, diffusion_rate)
        grids.append(grid.numpy())  # Store each step
    
    return grids

# Define the shape, mask, and initial heat source as before
shape = (100, 100)
mask = torch.zeros(shape)
mask[30:70, 30:70] = 1
initial_heat_source = (slice(40, 60), slice(40, 60))

# Simulate heat diffusion
grids = simulate_heat_diffusion(shape, mask, initial_heat_source, steps=100)

# Create an animation
fig = go.Figure(
    frames=[
        go.Frame(data=go.Heatmap(z=grid, colorscale='Hot'), name=str(k))
        for k, grid in enumerate(grids)
    ]
)

# Add initial data
fig.add_trace(go.Heatmap(z=grids[0], colorscale='Hot'))

# Layout configuration
fig.update_layout(
    title="Heat Diffusion Process",
    width=600,
    height=600,
    updatemenus=[{
        "type": "buttons",
        "buttons": [{
            "label": "Play",
            "method": "animate",
            "args": [None, {"frame": {"duration": 100, "redraw": True}, "fromcurrent": True}]
        },
        {
            "label": "Pause",
            "method": "animate",
            "args": [[None], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}]
        }]
    }]
)

# Slider configuration
fig.update_layout(sliders=[{
    "steps": [{"args": [[f.name], {"frame": {"duration": 100, "redraw": True}, "mode": "immediate"}], "label": str(k), "method": "animate"} for k, f in enumerate(fig.frames)]
}])

fig.show()
