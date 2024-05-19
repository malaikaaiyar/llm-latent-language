# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
# %%


def plot_heatmaps(data, metric='p_alt'):
    # Determine dimensionality of data (2D or 3D based on tuple size)
    is_3d = len(next(iter(data.keys()))) == 3
    
    # Extract unique x-values and y-values, and z-values if 3D
    x_values = sorted(set(k[0] for k in data.keys()))  # Start layers
    y_values = sorted(set(k[1] for k in data.keys()))  # End layers
    z_values = sorted(set(k[2] for k in data.keys())) if is_3d else [None]
    
    # Create subplots
    fig, axes = plt.subplots(1, len(z_values), figsize=(5 * max(1, len(z_values)), 5), sharey=True)
    if len(z_values) == 1:
        axes = [axes]  # Ensure axes is iterable

    # Initialize heatmap data storage
    heatmaps = {z: np.full((len(y_values), len(x_values)), np.nan) for z in z_values}

    # Populate heatmap data
    for key, metrics in data.items():
        x, y = key[0], key[1]
        if y < x+1:  # Skip invalid entries
            continue
        z = key[2] if is_3d else None
        x_idx = x_values.index(x)
        y_idx = y_values.index(y)
        heatmaps[z][y_idx, x_idx] = metrics.get(metric, np.nan)  # Extracting metric dynamically

    # Modify the colormap to display NaN values as light grey
    current_cmap = plt.get_cmap('magma').copy()
    current_cmap.set_bad(color='lightgrey')

    # Plot each heatmap
    for ax, z in zip(axes, z_values):
        sns.heatmap(heatmaps[z], ax=ax, cmap=current_cmap, cbar=(z == z_values[-1]), cbar_kws={"shrink": .8})
        ax.set_title(f'{metric=}, {z=}' if is_3d else f'{metric=}')
        ax.set_xlabel('Start Layer')
        ax.set_ylabel('End Layer')
        
        # Adjust tick marks to be centered in cells (at half-integer values)
        ax.set_xticks(np.arange(len(x_values)) + 0.5, minor=False)
        ax.set_yticks(np.arange(len(y_values)) + 0.5, minor=False)
        ax.set_xticklabels(x_values, fontsize=6)
        ax.set_yticklabels(y_values, fontsize=6)
        
        # Align grid lines with the data indices (integer values, major grid)
        ax.set_xticks(np.arange(len(x_values)), minor=True)
        ax.set_yticks(np.arange(len(y_values)), minor=True)
        ax.tick_params(which='minor', length=0)
        ax.grid(True, which='minor', color='grey', linestyle='-', linewidth=0.5)

    plt.tight_layout()
    plt.show()
    return heatmaps

def print_top_k_entries(data, sort_key, k=5, largest=True):
    """
    Print the top-k entries from a dictionary sorted by a specific key in descending order.

    Args:
    data (dict): The dictionary containing the data.
    k (int): The number of top entries to print.
    sort_key (str): The key to sort the entries by.

    """
    # Create a list from the dictionary, adding negative infinity where the sort key is missing
    sorted_list = sorted(data.items(), key=lambda item: item[1].get(sort_key, float('-inf')), reverse=largest)

    print(f"Top {k} entries sorted by '{sort_key}' in {'descending' if largest else 'ascending'} order:")
    print("=" * 30)
    # Print the top-k items
    for i, (key, values) in enumerate(sorted_list[:k]):
        print(f"Rank {i + 1}: {key} - {sort_key} = {values.get(sort_key)}")
    print("")

# Load the data from the pickle file
with open('out/hook_reject_alt_sweep.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    print(list(data_dict.values())[0].keys())

# Call the plot_heatmaps function with the loaded data
heatmaps = plot_heatmaps(data_dict, 'p_out')
print_top_k_entries(data_dict, 'p_out', largest=False)
print_top_k_entries(data_dict, 'lp_out', largest=False)

# %%
