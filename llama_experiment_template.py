# %%
from tracemalloc import start
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm import tqdm
import pickle 
import argparse
import pprint
# === Typing Libraries ===
from typing import Tuple, List, Optional, Dict, Callable, Iterable, Any
from jaxtyping import Int, Float
from beartype import beartype

# ==== Torch/Transformer Libraries ====
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
import gen_data
from utils import plot_ci_plus_heatmap
from tuned_lens_wrap import load_tuned_lens
from dq_utils import proj, entropy, plot_ci, is_chinese_char, measure_performance
from logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap
import intervention
from intervention import Intervention
from config_argparse import parse_args
# %%
@dataclass
class Config:
    seed: int = 42
    src_lang: str = 'fr'
    dest_lang: str = 'zh'
    latent_lang: str = 'en'
    model_size: str = '7b'
    model_name: str = 'meta-llama/Llama-2-%s-hf' % model_size
    single_token_only: bool = False
    multi_token_only: bool = False
    out_dir: str = './visuals'
    hf_token: str = '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'
    dataset_path: str = "./data/synth_llama2"
    debug: bool = True
    num_multi_shot : int = 5
    token_add_spaces: bool = True
    token_add_leading_byte: bool = False
    token_add_prefixes : bool = False
    dataset_filter_correct : bool = True
    use_tuned_lens : bool = True
    intervention_correct_latent_space : bool = True
    steer_scale_coeff : float = 1.0
    start_layer_low : int = 0
    start_layer_high : int = 32
    end_layer_low : int = 0
    end_layer_high : int = 32
    intervention_func : str = 'hook_reject_subspace'
    log_file : str = 'DUMMY_NAME'
    metric : str = 'p_alt'
    metric_goal : str = 'max'

cfg = Config()

try:
    # The get_ipython function is available in IPython environments
    ipython = get_ipython()
    if 'IPKernelApp' not in ipython.config:  # Check if not within an IPython kernel
        raise ImportError("Not in IPython")
    print("Enabling autoreload in IPython.")
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
except Exception as e:
    print(f"Not in an IPython environment: {e}")
    # Parse command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--log_file", type=str, default="experiment.log", help="File to write experiment log to")
    # cli_args = parser.parse_args()
    # print(f"Writing experiment log to {cli_args.log_file}")
    cfg = parse_args(cfg)
    #pprint.pprint(asdict(cfg))
    assert cfg.log_file != 'DUMMY_NAME', "ERROR: log_file not set"
cfg_dict = asdict(cfg)
# %%
# fix random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_grad_enabled(False)
# %%
pd.set_option('display.max_rows', 100)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect the display width for wrapping
pd.set_option('display.max_colwidth', None)  # Show full length of data in columns

# %%

    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


# tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
# tokenizer_vocab = tokenizer.get_vocab()
# %%
if 'LOAD_MODEL' not in globals():
    LOAD_MODEL = False
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype = torch.float16)
    tokenizer_vocab = model.tokenizer.get_vocab()
    if cfg.use_tuned_lens:
        tuned_lens = load_tuned_lens(model).float()
        model.tuned_lens = tuned_lens
# %%
# df_src = pd.read_csv(os.path.join(cfg.dataset_path, cfg.src_lang, 'clean.csv')).reindex()
# df_dest = pd.read_csv(os.path.join(cfg.dataset_path, cfg.dest_lang, 'clean.csv')).reindex()
# df_raw_data = gen_data.merge_datasets(df_src, df_dest, tokenizer_vocab, cfg)
df_raw_data = pd.read_csv(os.path.join(cfg.dataset_path, 'llama2_all_no_space.csv'))
df_raw_data = gen_data.filter_matching_translations(df_raw_data)
dataset = gen_data.gen_translation_task(df_raw_data, tokenizer_vocab, **cfg_dict)
correct_dataset = gen_data.filter_correct(dataset, model)
print(dataset[0]['prompt'])
#hf_model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, load_in_8bit=True)
measure_performance(correct_dataset, model)
# %%

layer_log2 = {}

start_lower, start_upper = cfg.start_layer_low, cfg.start_layer_high
end_lower, end_upper = cfg.end_layer_low, cfg.end_layer_high
#steer_scale_coeff_list = [0.9, 1.0, 1.01, 1.02, 1.05, 1.3, 1.5][::-1]
#steer_scale_coeff_list = [1.0]


def calculate_iterations(start_lower, start_upper, end_lower, end_upper):
    if start_upper <= start_lower or end_upper <= end_lower:
        return 0  # No valid iterations if ranges are non-positive or improperly defined

    # Maximum valid start_layer is start_upper - 1
    # Minimum valid end_layer is start_layer + 1, which translates to start_lower + 1 for start_lower
    if end_upper <= start_lower + 1:
        return 0  # No valid end_layer values if end_upper is less than or equal to start_lower + 1

    # Applying the formula: Summing (end_upper - k - 1) for k from start_lower to start_upper - 1
    total_iterations = 0
    for k in range(start_lower, start_upper):
        if k + 1 < end_upper:  # Ensure that there is at least one valid end_layer
            total_iterations += (end_upper - (k + 1))

    return total_iterations
total_iterations = calculate_iterations(start_lower, start_upper, end_lower, end_upper)
outer_pbar = tqdm(total=total_iterations, desc='Overall Progress', leave=True)

import intervention
from logit_lens import get_logits, plot_logit_lens_latents

def format_dict_single_line_custom(d):
    # Create a formatted string from dictionary entries
    items = [f"{k}: {f'{v:.4f}' if isinstance(v, float) else v}" for k, v in d.items()]
    # Join all items in a single line
    return ', '.join(items)

def is_better(stats, best_stats, cfg):
    if cfg.metric_goal == 'max':
        return stats[cfg.metric] > best_stats[cfg.metric]
    else:
        return stats[cfg.metric] < best_stats[cfg.metric]

if cfg.metric_goal == 'max':
    best_stats = {cfg.metric: -np.inf}
else:
    best_stats = {cfg.metric: np.inf}
    
for start_layer in range(start_lower,start_upper):
    for end_layer in range(end_lower, end_upper):
        if start_layer >= end_layer:
            continue
        
        intervene_diff = Intervention(cfg.intervention_func, range(start_layer, end_layer))
        latent_diff, logits_diff = get_logits(correct_dataset, model, intervention=intervene_diff,  **cfg_dict)

        stats = plot_logit_lens_latents(logits_diff, correct_dataset, **cfg_dict, title="diff", cfg=cfg, only_compute_stats=False)
        
        if is_better(stats, best_stats, cfg):
            new_best_msg = f"New best stats: start_layer={start_layer}, end_layer={end_layer}, {format_dict_single_line_custom(stats)}"
            tqdm.write(new_best_msg)  # Using tqdm.write to avoid interference with the progress bar
            outer_pbar.set_description(f"")
            best_stats = stats
        else:
            outer_pbar.set_description(f"Best: {format_dict_single_line_custom(best_stats)} + Current: {format_dict_single_line_custom(stats)}")
        outer_pbar.update(1)  # Increment the progress bar after each inner iteration
        layer_log2[(start_layer, end_layer)] = stats


outer_pbar.close()  # Ensure to close the progress bar after the loop completes
# Save layer_log2 to a pickle file
pickle.dump(layer_log2, open(cfg.log_file + ".pkl", "wb"))

log_legend = """
Measuring 
lp_out/p_out : logprobs/probs of correct answer
lp_alt/p_alt logprobs/probs of alternate answer
lp_diff/p_ratio: logprob_diff/probs ration of alt-correct or alt/correct
"""

pp = pprint.PrettyPrinter(sort_dicts=False)
# Save log_legend to the log file
with open(cfg.log_file + ".log", "a") as f:
    f.write("Command: " + ' '.join(sys.argv) + "\n")
    f.write(pp.pformat(asdict(cfg)))
    f.write("\n==============\n")
    f.write(intervene_diff.description)
    f.write("\n==============\n")
    f.write(log_legend)

print("Done!")

# %%
