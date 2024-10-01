# %%
# %load_ext autoreload
# %autoreload 2
# %%
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# %%
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt

# %%
from utils.config_argparse import try_parse_args
# %%

# ==== Custom Libraries ====

# %%
@dataclass
class Config:
    seed: int = 42
    model_name: str = "meta-llama/Llama-2-7b-hf"
    # single_token_only: bool = False
    # multi_token_only: bool = False
    out_dir: str = './out_iclr'

cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
    
os.makedirs(cfg.out_dir, exist_ok=True)


# %%

def read_results_csv(file_path):
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"File not found: {file_path}")
        return pd.DataFrame()

# Assuming cfg and short_model_name are defined earlier in your code


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def plot_language_performance(df, filename = None, group_by_target=False, cfg=None):
    # Pivot the dataframe to have dest_langs as columns and src_langs as index
    pivot_df = df.pivot(index='src_lang', columns='dest_lang', values=['avg', 'sem95_error'])
    
    # Get source and destination languages
    src_langs = pivot_df.index
    dest_langs = pivot_df.columns.get_level_values(1).unique()

    # Determine primary and secondary languages based on grouping
    if group_by_target:
        primary_langs = dest_langs
        secondary_langs = src_langs
        x_label = 'Target Language'
        legend_prefix = 'from'
    else:
        primary_langs = src_langs
        secondary_langs = dest_langs
        x_label = 'Source Language'
        legend_prefix = 'to'

    # Set up the plot
    fig, ax = plt.subplots(figsize=(8, 4))

    # Set width of bars
    bar_width = 0.15
    
    # Generate a color palette based on the number of secondary languages
    colors = sns.color_palette(n_colors=len(secondary_langs))

    # Create a dictionary to store the bar objects for the legend
    legend_elements = {}

    # Plot bars for each language pair
    for i, primary_lang in enumerate(primary_langs):
        x_offset = 0
        for j, secondary_lang in enumerate(secondary_langs):
            if primary_lang != secondary_lang:  # Skip same language pairs
                if group_by_target:
                    height = pivot_df['avg'][primary_lang][secondary_lang]
                    yerr = pivot_df['sem95_error'][primary_lang][secondary_lang]
                else:
                    height = pivot_df['avg'][secondary_lang][primary_lang]
                    yerr = pivot_df['sem95_error'][secondary_lang][primary_lang]
                
                if not pd.isna(height):
                    x = i + x_offset * bar_width
                    bar = ax.bar(x, height, bar_width, yerr=yerr, capsize=5, 
                                 color=colors[j], alpha=0.8)
                    
                    # Store the bar object for the legend
                    if secondary_lang not in legend_elements:
                        legend_elements[secondary_lang] = bar[0]
                    
                    x_offset += 1

    # Customize the plot
    ax.set_ylabel('Probability of Correct Translation')
    ax.set_xlabel(x_label)
    ax.set_title('Language Translation Performance')
    
    # Set xticks with offset
    xtick_positions = np.arange(len(primary_langs)) + 2 * bar_width
    ax.set_xticks(xtick_positions)
    ax.set_xticklabels(primary_langs)

    # Create the legend using the stored bar objects
    ax.legend(legend_elements.values(), [f'{legend_prefix} {lang.upper()}' for lang in legend_elements.keys()])

    # Set y-axis to start at 0 and end at 1
    ax.set_ylim(0, 1)

    # Add a grid for better readability
    ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    # Adjust layout and display the plot
    plt.tight_layout()

    # Save the plot as SVG
    if cfg:
        out_dir = os.path.join(cfg.out_dir, 'plots', cfg.model_name.split("/")[-1])
        os.makedirs(out_dir, exist_ok=True)
        file_name = 'translation.svg' if not filename else filename
        file_path = os.path.join(out_dir, file_name)
        plt.savefig(file_path, format='svg')
        print(f"Plot saved as {file_path}")

    plt.show()

# Assuming df_no_interv is already loaded with the data
# Call the function to generate the plot
# For grouping by source language (default):

# %%


def plot_language_performance_grid(df, filename = None, group_by_target=False, fig_size = (6,5), cfg=None):
    # Get unique latent languages
    latent_langs = df['latent_lang'].unique()
    
    # Set up the plot grid
    n_plots = len(latent_langs)
    n_cols = min(3, n_plots)  # Max 3 columns
    n_rows = (n_plots - 1) // n_cols + 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_size[0]*n_cols, fig_size[1]*n_rows), squeeze=False)
    
    # Flatten axes array for easy iteration
    axes_flat = axes.flatten()

    for idx, latent_lang in enumerate(latent_langs):
        ax = axes_flat[idx]
        
        # Filter data for current latent language
        df_latent = df[df['latent_lang'] == latent_lang]
        
        # Pivot the dataframe
        pivot_df = df_latent.pivot(index='src_lang', columns='dest_lang', values=['avg', 'sem95_error'])
        
        # Get source and destination languages
        src_langs = pivot_df.index.unique()
        dest_langs = pivot_df.columns.get_level_values(1).unique()

        # Determine primary and secondary languages based on grouping
        if group_by_target:
            primary_langs = dest_langs
            secondary_langs = src_langs
            x_label = 'Target Language'
            legend_prefix = 'from'
        else:
            primary_langs = src_langs
            secondary_langs = dest_langs
            x_label = 'Source Language'
            legend_prefix = 'to'

        # Set width of bars
        bar_width = 0.15
        
        # Generate a color palette
        colors = sns.color_palette(n_colors=len(secondary_langs))

        # Plot bars for each language pair
        for i, primary_lang in enumerate(primary_langs):
            x_offset = 0
            for j, secondary_lang in enumerate(secondary_langs):
                if primary_lang != secondary_lang:  # Skip same language pairs
                    if group_by_target:
                        height = pivot_df['avg'][primary_lang][secondary_lang]
                        yerr = pivot_df['sem95_error'][primary_lang][secondary_lang]
                    else:
                        height = pivot_df['avg'][secondary_lang][primary_lang]
                        yerr = pivot_df['sem95_error'][secondary_lang][primary_lang]
                    
                    if not pd.isna(height):
                        x = i + x_offset * bar_width
                        ax.bar(x, height, bar_width, yerr=yerr, capsize=5, 
                               color=colors[j], alpha=0.8)
                        x_offset += 1

        # Customize the subplot
        ax.set_ylabel('Probability of Correct Translation')
        ax.set_xlabel(x_label)
        ax.set_title(f'Latent Language: {latent_lang.upper()}')
        
        # Set xticks with offset
        xtick_positions = np.arange(len(primary_langs)) + bar_width * 1.5
        ax.set_xticks(xtick_positions)
        ax.set_xticklabels(primary_langs)

        # Set y-axis to start at 0 and end at 1
        ax.set_ylim(0, 1)

        # Add a grid for better readability
        ax.yaxis.grid(True, linestyle='--', which='major', color='grey', alpha=.25)

    # Create a common legend
    handles = [plt.Rectangle((0,0),1,1, color=colors[i]) for i in range(len(secondary_langs))]
    fig.legend(handles, [f'{legend_prefix} {lang.upper()}' for lang in secondary_langs], 
               loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=len(secondary_langs))

    # Remove any unused subplots
    for j in range(idx+1, len(axes_flat)):
        fig.delaxes(axes_flat[j])

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot as SVG
    if cfg:
        out_dir = os.path.join(cfg.out_dir, 'plots', cfg.model_name.split("/")[-1])
        os.makedirs(out_dir, exist_ok=True)
        file_name = 'translation_grid.svg' if not filename else filename
        file_path = os.path.join(out_dir, file_name)
        plt.savefig(file_path, format='svg')
        print(f"Plot saved as {file_path}")

    plt.show()

# Assuming df_interv is loaded with the data from translation_interv.csv
# Call the function to generate the plot

def run(short_model_name, out_dir = './out_iclr'):

    cfg = Config(out_dir=out_dir, model_name=short_model_name)
    # Define file paths
    no_interv_path = os.path.join(out_dir, short_model_name, "translation_no_interv.csv")
    interv_path = os.path.join(out_dir, short_model_name, "translation_interv.csv")
    interv_alt_path = os.path.join(out_dir, short_model_name, "translation_interv_alt.csv")

    # Read CSV files and create DataFrames
    df_no_interv = read_results_csv(no_interv_path)
    df_interv = read_results_csv(interv_path)
    df_interv_alt = read_results_csv(interv_alt_path)


    print("No Intervention performance")
    plot_language_performance(df_no_interv, group_by_target=False, filename='translation_no_interv.svg', cfg=cfg)
    print("Intervention performance = reject correct word")
    plot_language_performance_grid(df_interv, fig_size = (5,3), filename='translation_interv_grid.svg', cfg=cfg)
    print("Intervention performance = reject random word")
    plot_language_performance_grid(df_interv_alt, fig_size = (5,3), filename='translation_interv_alt_grid.svg', cfg=cfg)
# For grouping by target language:
# plot_language_performance_grid(df_interv, group_by_target=True)
# %%
# %%
if cfg.model_name == 'all':
    for short_model_name in ['gemma-2-9b', 'gemma-2-2b', 'Llama-2-7b-hf', 'Llama-2-13b-hf']:
        run(short_model_name)
else:
    short_model_name = cfg.model_name.split("/")[-1]
    run(short_model_name)
# %%