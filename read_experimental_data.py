# %%
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 
import re
from collections import defaultdict
import itertools
import warnings
# %%

def plot_heatmap_on_ax(data, ax, metric='p_alt', z_value=None, vmin=None, vmax=None):
    # Determine if the data includes a z-value (3D)
    is_3d = len(next(iter(data.keys()))) == 3
    
    # Extract unique x-values and y-values
    x_values = sorted(set(k[0] for k in data.keys()))
    y_values = sorted(set(k[1] for k in data.keys()))
    
    # Initialize heatmap data storage
    heatmap_data = np.full((len(y_values), len(x_values)), np.nan)
    
    # Populate heatmap data
    for key, metrics in data.items():
        x, y = key[0], key[1]
        if y < x+1:  # Skip invalid entries
            continue
        if is_3d:
            z = key[2]
            if z != z_value:
                continue
        x_idx = x_values.index(x)
        y_idx = y_values.index(y)
        heatmap_data[y_idx, x_idx] = metrics.get(metric, np.nan)  # Extracting metric dynamically

    # Check if vmin and vmax are provided, else compute from data
    if vmin is None or vmax is None:
        vmin = np.nanmin(heatmap_data)
        vmax = np.nanmax(heatmap_data)

    # Modify the colormap to display NaN values as light grey
    current_cmap = plt.get_cmap('magma').copy()
    current_cmap.set_bad(color='lightgrey')

    # Plot heatmap on the provided ax object
    sns.heatmap(heatmap_data, ax=ax, cmap=current_cmap, cbar=True, vmin=vmin, vmax=vmax, cbar_kws={"shrink": .8})
    ax.set_title(f'{metric=}, z={z_value}' if is_3d else f'{metric=}')
    ax.set_xlabel('Start Layer')
    ax.set_ylabel('End Layer')
    
    # Adjust tick marks to be centered in cells
    ax.set_xticks(np.arange(len(x_values)) + 0.5, minor=False)
    ax.set_yticks(np.arange(len(y_values)) + 0.5, minor=False)
    ax.set_xticklabels(x_values, fontsize=6)
    ax.set_yticklabels(y_values, fontsize=6)
    
    # Align grid lines with the data indices
    ax.set_xticks(np.arange(len(x_values)), minor=True)
    ax.set_yticks(np.arange(len(y_values)), minor=True)
    ax.tick_params(which='minor', length=0)
    ax.grid(True, which='minor', color='grey', linestyle='-', linewidth=0.5)
    

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
with open('out/hook_reject_layer_search.pkl', 'rb') as f:
    data_dict = pickle.load(f)
    print(list(data_dict.values())[0].keys())

# Call the plot_heatmaps function with the loaded data
# heatmaps = plot_heatmaps(data_dict, 'p_out')
# print_top_k_entries(data_dict, 'p_out', largest=False)
# print_top_k_entries(data_dict, 'lp_out', largest=False)

# %%
def plot_files_in_folder(folder_path, 
                         plot_function, 
                         metric='p_alt', 
                         z_value=None, 
                         filename_pattern=r'hook_reject_(.*)_en_(.*).pkl', 
                         cols=4, 
                         img_path = None,
                         title_func=lambda x: x, 
                         colour_range = None,
                         order = lambda x: x):
    # Compile the regex pattern to match filenames
    regex = re.compile(filename_pattern)
    
    # List only files that match the regex pattern in the given folder
    files = [f for f in os.listdir(folder_path) if re.match(regex, f) and f.endswith('.pkl') and os.path.isfile(os.path.join(folder_path, f))]
    
    # Determine the number of matching files
    num_files = len(files)
    
    # Check if there are any files to process
    if num_files == 0:
        print("No matching .pkl files found in the directory.")
        return
    
    # Determine grid size for plotting
    cols = min(cols, num_files)  # No more than 4 columns
    rows = (num_files + cols - 1) // cols  # Calculate rows needed
    
    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    axes = axes.flatten()  # Flatten axes array for easy iteration
    
    if colour_range:
        vmin, vmax = colour_range
    else:
        vmin, vmax = None, None
    
    # Plot each matching .pkl file on its corresponding axes
    for i, filename in enumerate(order(files)):
        file_path = os.path.join(folder_path, filename)
        
        # Load data from .pkl file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        
        # Use the provided plotting function on each subplot
        plot_function(data, axes[i], metric=metric, z_value=z_value, vmin=vmin, vmax=vmax)
        
        # Set the title of each subplot to the filename
        title_name = title_func(filename)
        axes[i].set_title(title_name)
    
    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    # Save the figure as SVG to the specified image path
    if img_path:
        fig.savefig(img_path, format='svg')

def group_and_interleave_filenames(filenames):
    # Compile the regex pattern
    pattern = re.compile(r'hook_reject_(.+)_(.+)_(.+)\.pkl')
    
    # Dictionary to hold lists of filenames grouped by the last entry
    grouped_files = defaultdict(list)
    
    # Group filenames by the second entry
    for filename in filenames:
        match = pattern.match(filename)
        if match:
            second = match.group(2)  # Get the second entry from the regex match
            grouped_files[second].append(filename)
    
    # Check if all groups are the same size
    sizes = {len(v) for v in grouped_files.values()}
    if len(sizes) > 1:
        warnings.warn("Warning: Not all groups are the same size.")

    # Interleave groups
    interleaved_result = list(itertools.chain.from_iterable(itertools.zip_longest(*grouped_files.values())))
    
    # Remove None entries if groups are uneven
    interleaved_result = [item for item in interleaved_result if item is not None]
    
    return interleaved_result


def format_filename(filename):
    # Define the regex pattern to capture the three groups in the filename
    pattern = r'hook_reject_(.+)_(.+)_(.+)\.pkl'
    
    # Use regex to match the pattern and capture the groups
    match = re.match(pattern, filename)
    
    # Check if the match was successful
    if match:
        # Extract the groups
        source, latent, target = match.groups()
        
        # Return the formatted string
        return f"Source: {source} Latent: {latent} Target: {target}"
    else:
        # Return an error message if the pattern does not match
        return "Invalid filename format"

# %%
def plot_multiple_files(file_paths, plot_function, metric='p_alt', z_value=None, vmin=None, vmax=None, cols=3, titles = None, img_path = None, color_range = None):
    """
    Plots a grid of heatmaps from a list of .pkl files.

    Args:
    file_paths (list): List of paths to .pkl files.
    plot_function (callable): Function to use for plotting, e.g., plot_heatmap_on_ax.
    metric (str): Metric to plot.
    z_value (optional): Specific z-value to filter on if data is 3D.
    vmin (float, optional): Minimum value for color scaling.
    vmax (float, optional): Maximum value for color scaling.
    cols (int): Number of columns in the subplot grid.
    """
    num_files = len(file_paths)
    cols = min(cols, num_files)  # No more than 4 columns
    rows = (num_files + cols - 1) // cols  # Calculate rows needed


    if titles is None:
        titles = [None] * num_files

    # Create a figure with subplots
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5), squeeze=False)
    axes = axes.flatten()  # Flatten axes array for easy iteration

    # Iterate over each file and subplot
    for i, (file_path, title) in enumerate(zip(file_paths, titles)):
        # Load data from .pkl file
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        if color_range:
            vmin, vmax = color_range

        # Use the provided plotting function on each subplot
        plot_function(data, axes[i], metric=metric, z_value=z_value, vmin=vmin, vmax=vmax)

        # Set the title of each subplot to the filename
        axes[i].set_title(title)

    # Hide unused axes if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    fig.savefig(img_path, format='svg')
    

# # Example usage:
# paths = ['out/reject_lang_sweep/hook_reject_fr_en_zh.pkl']
# plot_multiple_files(paths, plot_heatmap_on_ax, metric='p_out', img_path = 'icml-llama-english/img/hook_reject_fr_en_zh.svg')
# # %%
# paths = ['out/reject_lang_sweep/hook_reject_fr_en_zh.pkl']
# plot_multiple_files(paths, plot_heatmap_on_ax, metric='p_out', img_path = 'icml-llama-english/img/hook_reject_fr_en_zh.svg')

# # %%

# paths = ['out/hook_only_new_subspace_alt/zh_enalt_fr_50.pkl']
# plot_multiple_files(paths, plot_heatmap_on_ax, metric='p_out', img_path = 'icml-llama-english/img/zh_enalt_fr_50_p_out.svg')


# plot_multiple_files(paths, plot_heatmap_on_ax, metric='p_alt', img_path = 'icml-llama-english/img/zh_enalt_fr_50_p_alt.svg')


# plot_multiple_files(paths, plot_heatmap_on_ax, metric='lp_diff', img_path = 'icml-llama-english/img/zh_enalt_fr_50_lp_diff.svg')

# # %%
# plot_files_in_folder('out/reject_lang_sweep', plot_heatmap_on_ax, metric = 'p_out', filename_pattern='hook_reject_(\w+)_en_(\w+).pkl', cols=3, img_path = 'img/hook_reject_en_latent_sweep2.svg',
#                      title_func=format_filename, order= group_and_interleave_filenames)

# # %%
# plot_files_in_folder('out/reject_lang_sweep', plot_heatmap_on_ax, metric = 'lp_out', filename_pattern='hook_reject_(\w+)_(\w+)_(\w+).pkl', cols=4, img_path = 'img/hook_reject_all_latent_sweep_lp_out_range.svg',
#                      title_func=format_filename, order = group_and_interleave_filenames, colour_range=(-8, 0))
# # %%
# plot_files_in_folder('out/reject_lang_sweep', plot_heatmap_on_ax, metric = 'p_out', filename_pattern='hook_reject_(\w+)_en_(\w+).pkl', cols=3, img_path = 'img/hook_reject_enalt_latent_sweep.svg',
#                      title_func=format_filename, order= group_and_interleave_filenames)
# plot_files_in_folder('out/reject_lang_sweep', plot_heatmap_on_ax, metric = 'lp_out', filename_pattern='hook_reject_(\w+)_en_(\w+).pkl', cols=3, img_path = 'img/hook_reject_enalt_latent_sweep_lp.svg',
#                      title_func=format_filename, order= group_and_interleave_filenames)
# # %%
# plot_files_in_folder('out/reject_lang_sweep_alt', plot_heatmap_on_ax, metric = 'lp_out', filename_pattern='hook_reject_(\w+)_(\w+)_(\w+).pkl', cols=4, img_path = 'img/hook_reject_alt_latent_sweep_lp.svg',
#                      title_func=format_filename, order= group_and_interleave_filenames)


# plot_files_in_folder('out/reject_lang_sweep_alt', plot_heatmap_on_ax, metric = 'p_out', filename_pattern='hook_reject_(\w+)_(\w+)_(\w+).pkl', cols=4, img_path = 'img/hook_reject_alt_latent_sweep.svg',
#                      title_func=format_filename, order= group_and_interleave_filenames)
# %%


def format_filename2(filename):
    # Define the regex pattern to capture the three groups in the filename
    pattern = r'(.+)_(.+)_(.+)_(.+)\.pkl'
    
    # Use regex to match the pattern and capture the groups
    match = re.match(pattern, filename)
    
    # Check if the match was successful
    if match:
        # Extract the groups
        source, latent, target, c = match.groups()
        
        # Return the formatted string
        return f"Source: {source} Latent: {latent} Target: {target} c={(float(c) / 10):.02f}"
    else:
        # Return an error message if the pattern does not match
        return "Invalid filename format"

# %%

# plot_files_in_folder('out/hook_only_new_subspace_alt', plot_heatmap_on_ax, metric = 'lp_alt', filename_pattern='(\w+)_(\w+)alt_(\w+)_(\d)+.pkl', cols=4, img_path = 'img/hook_only_new_subspace_alt_lp.svg',
#                      title_func=format_filename2, order=sorted)

# plot_files_in_folder('out/hook_only_new_subspace_alt', plot_heatmap_on_ax, metric = 'p_alt', filename_pattern='(\w+)_(\w+)alt_(\w+)_(\d)+.pkl', cols=4, img_path = 'img/hook_only_new_subspace_alt_p.svg',
#                      title_func=format_filename2, order=sorted)

# plot_files_in_folder('out/hook_only_new_subspace_alt', plot_heatmap_on_ax, metric = 'lp_diff', filename_pattern='(\w+)_(\w+)alt_(\w+)_(\d)+.pkl', cols=4, img_path = 'img/hook_only_new_subspace_alt_diff.svg',
#                      title_func=format_filename2, order=sorted)
# # %%
# plot_files_in_folder('out/hook_only_new_subspace_alt', plot_heatmap_on_ax, metric = 'p_out', filename_pattern='(\w+)_(\w+)alt_(\w+)_(\d)+.pkl', cols=4, img_path = 'img/hook_only_new_subspace_p.svg',
#                      title_func=format_filename2, order=sorted)
# # %%
# plot_files_in_folder('out/hook_only_new_subspace', plot_heatmap_on_ax, metric = 'p_out', filename_pattern='(\w+)_(\w+)_(\w+)_(\d)+.pkl', cols=4, img_path = 'img/hook_only_new_subspace_p.svg',
#                      title_func=format_filename2, order=sorted)
# %%

for metric in ['p_out', 'p_alt']:

    plot_files_in_folder('out/hook_only_new_subspace', plot_heatmap_on_ax, metric = metric, filename_pattern='zh_en_fr_(\d)+.pkl', cols=5, img_path = f'img/steer_zh_en_fr_{metric}.svg',
                        title_func=format_filename2, order=sorted)
# %%
for metric in ['p_out', 'p_alt']:

    plot_files_in_folder('out/hook_only_new_subspace_alt', plot_heatmap_on_ax, metric = metric, filename_pattern='zh_enalt_fr_(\d)+.pkl', cols=5, img_path = f'img/steer_zh_enalt_fr_{metric}.svg',
                        title_func=format_filename2, order=sorted)
# %%
for metric in ['p_out', 'p_alt']:
    plot_files_in_folder('out/hook_only_new_subspace', plot_heatmap_on_ax, metric = metric, filename_pattern='zh_de_fr_(\d)+.pkl', cols=5, img_path = f'img/steer_zh_de_fr_{metric}.svg',
                        title_func=format_filename2, order=sorted)
# %%
