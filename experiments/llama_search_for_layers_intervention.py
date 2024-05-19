# %%
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

# %%
from tracemalloc import start
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm import tqdm
import pickle 

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
    steer_scale_coeff : float = 1.0
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cfg = Config()
cfg_dict = asdict(cfg)
# tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
# tokenizer_vocab = tokenizer.get_vocab()
# %%
if 'LOAD_MODEL' not in globals():
    LOAD_MODEL = False
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype = torch.float16)
    tuned_lens = load_tuned_lens(model).float()
    tokenizer_vocab = model.tokenizer.get_vocab()
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
# from intervention import Intervention
# import intervention 


# intervene_move = Intervention(intervention.hook_move_subspace, range(32))
# intervene_proj2 = Intervention(intervention.hook_move_subspace2, range(32))

# %%
#latent_probs, out_probs, entropy = measure_lang_probs(dataset)
# EXPENSIVE!
# from logit_lens import get_logits, compute_layer_probs#, plot_logit_lens_latents


# from tqdm import tqdm
# import pickle

# layer_log = {}

# total_iterations = sum(31 - i for i in range(32))  # Calculating the total number of iterations
# outer_pbar = tqdm(total=total_iterations, desc='Overall Progress', leave=True)

# best_ld_diff = -1000000
# for start_layer in range(32):
#     for end_layer in range(start_layer + 1, 32):
#         intervene_diff = Intervention(intervention.hook_diff_subspace, range(start_layer, end_layer))
#         latent_diff, logits_diff = get_logits(correct_dataset, model, intervention=intervene_diff, **cfg_dict)

#         stats = plot_logit_lens_latents(logits_diff, correct_dataset, **cfg_dict, title="diff", cfg=cfg, only_compute_stats=True)
        
#         if stats['lp_diff'] > best_ld_diff:
#             new_best_msg = f"New best: start_layer={start_layer}, end_layer={end_layer}, p_alt={stats['p_alt']:.04f}, ld_diff={stats['lp_diff']:.04f}"
#             tqdm.write(new_best_msg)  # Using tqdm.write to avoid interference with the progress bar
#             outer_pbar.set_description(f"Best prob {stats['p_alt']:.3f} ld_diff {stats['lp_diff']:.3f} layers {start_layer}->{end_layer}")
#             best_prob = stats['p_alt']
#             best_diff = stats['lp_diff']
#             best_start_layer = start_layer
#             best_end_layer = end_layer
#             best_stats = stats
#             best_ld_diff = stats['lp_diff']
        
#         outer_pbar.update(1)  # Increment the progress bar after each inner iteration
#         layer_log[(start_layer, end_layer)] = stats


# outer_pbar.close()  # Ensure to close the progress bar after the loop completes
# %%
# Save layer_log to a pickle file
# import pickle
# pickle.dump(layer_log, open('out/layer_intervention.pkl', 'wb'))
# %%

    
#plot_heatmaps(layer_log)
# %%
# p_alt = np.full((32, 32), -1000.0)
# lp_diff = np.full((32, 32), -1000.0)

# # Fill the arrays with data
# for (i, j), stats in layer_log.items():
#     p_alt[j, i] = stats['p_alt']
#     lp_diff[j, i] = stats['lp_diff']
# %%

# %%
# for start_layer in range(0, 32):
#     for end_layer in range(start_layer+1, 32):
#         best_prob = 0
#         intervene_diff = Intervention(intervention.hook_diff_subspace, range(start_layer,end_layer))
#         latent_diff, logits_diff = get_logits(correct_dataset, model, intervention=intervene_diff, **cfg_dict)
#         #latents, logits = get_logits(correct_dataset, model, intervention=None, cfg=cfg)
#         #latent_reject, logits_reject = get_logits(correct_dataset, model, intervention=intervene_reject, cfg=cfg)
#         def analysis(latents, logits, **kwargs):
#             stats = plot_logit_lens_latents(logits, correct_dataset, cfg=cfg, **kwargs)
#             latent_heatmap(latents.cpu().abs(), **kwargs)
#             return stats
#         stats = analysis(latent_diff, logits_diff, **cfg_dict, title="No intevention")
        
#         if stats['p_alt'] > best_prob:
#             print(f"New best: {start_layer=}, {end_layer=}, {stats}")
#             best_prob = stats['p_alt']
#             best_start_layer = start_layer
#             best_end_layer = end_layer
#             best_stats = stats
        
#analysis(latent_reject, logits_reject, **cfg_dict, title="Rejection")
# %%

layer_log2 = {}



start_lower, start_upper = 16,22
end_lower, end_upper = 27, 32
steer_scale_coeff_list = [0.9, 1.0, 1.01, 1.02, 1.05, 1.3, 1.5][::-1]

total_iterations = (start_upper - start_lower) * (end_upper - end_lower) * len(steer_scale_coeff_list)
outer_pbar = tqdm(total=total_iterations, desc='Overall Progress', leave=True)

import intervention
from logit_lens import get_logits, plot_logit_lens_latents

best_prob = 0
for steer_scale_coeff in steer_scale_coeff_list:
    for start_layer in range(start_lower,start_upper):
        for end_layer in range(end_lower, end_upper):
            intervene_diff = Intervention(intervention.hook_diff_subspace, range(start_layer, end_layer))
            cfg_dict['steer_scale_coeff'] = steer_scale_coeff
            latent_diff, logits_diff = get_logits(correct_dataset, model, intervention=intervene_diff,  **cfg_dict)

            stats = plot_logit_lens_latents(logits_diff, correct_dataset, **cfg_dict, title="diff", cfg=cfg, only_compute_stats=False)
            
            if stats['p_alt'] > best_prob:
                new_best_msg = f"New best: start_layer={start_layer}, end_layer={end_layer}, steer={steer_scale_coeff:.02f} p_alt={stats['p_alt']:.04f}, ld_diff={stats['lp_diff']:.04f}"
                tqdm.write(new_best_msg)  # Using tqdm.write to avoid interference with the progress bar
                outer_pbar.set_description(f"Best prob {stats['p_alt']:.3f} ld_diff {stats['lp_diff']:.3f} layers {start_layer}->{end_layer} steer_scale_coeff={steer_scale_coeff:.02f}")
                best_prob = stats['p_alt']
                best_diff = stats['lp_diff']
                best_start_layer = start_layer
                best_end_layer = end_layer
                best_stats = stats
                best_ld_diff = stats['lp_diff']
            else:
                tqdm.write(f"start_layer={start_layer}, end_layer={end_layer}, steer={steer_scale_coeff:.02f} p_alt={stats['p_alt']:.04f}, ld_diff={stats['lp_diff']:.04f}")
            
            outer_pbar.update(1)  # Increment the progress bar after each inner iteration
            layer_log2[(start_layer, end_layer, steer_scale_coeff)] = stats


outer_pbar.close()  # Ensure to close the progress bar after the loop completes
pickle.dump(layer_log2, open('out/layer_intervention4.pkl', 'wb'))
print("Wrote layer_intervention4.pkl")
# %%
