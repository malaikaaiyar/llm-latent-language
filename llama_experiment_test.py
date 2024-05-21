# %%
from tracemalloc import start
from more_itertools import only
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm.auto import tqdm
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
from reverse_tuned_lens import ReverseLens
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
    use_reverse_lens : bool = False
    rev_lens_scale : bool = 1
    only_compute_stats : bool = False

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
try: 
    ipython = get_ipython()
    # if in jupyter notebook, force variables
    cfg.use_reverse_lens = True
    
except:
    pass

if 'LOAD_MODEL' not in globals():
    LOAD_MODEL = False
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype = torch.float16)
    tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore    
    if cfg.use_tuned_lens or cfg.use_reverse_lens:
        tuned_lens = load_tuned_lens(model)
        model.tuned_lens = tuned_lens
    if cfg.use_reverse_lens:
        reverse_lens = ReverseLens.from_tuned_lens(tuned_lens)
        model.reverse_lens = reverse_lens
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

torch.cuda.empty_cache()

layer_log2 = {}
# %%
scales = []
for d in tqdm(dataset):
    tok = model.tokenizer.encode(d['prompt'], return_tensors="pt").to(device)
    output, cache = model.run_with_cache(tok, names_filter = ['ln_final.hook_scale'])
    scales.append(cache['ln_final.hook_scale'])

# %%



from logit_lens import plot_logit_lens_latents
layers_range = range(20, 31)
# %%
cfg.use_reverse_lens = False
cfg.intervention_func = "hook_diff_subspace"
cfg_dict = asdict(cfg)
intervene_diff = Intervention(cfg.intervention_func, layers_range)
latent_diff, logits_diff = get_logits(correct_dataset, model, intervention=intervene_diff,  **cfg_dict)
latent_diff = latent_diff.float()
logits_diff = logits_diff.float()
stats = plot_logit_lens_latents(logits_diff, correct_dataset, only_compute_stats = False, **cfg_dict, title="diff", cfg=cfg, )
latent_heatmap(latent_diff, **cfg_dict, title="diff", cfg=cfg)
# %%
from logit_lens import plot_logit_lens_latents, get_logits, latent_heatmap
best_rev = {}
for steer_scale_coeff in torch.arange(0,1,0.1):
    rev_lens_scale = 2
    cfg.use_reverse_lens = True
    cfg.rev_lens_scale = rev_lens_scale
    cfg.steer_scale_coeff = steer_scale_coeff
    cfg.intervention_func = "hook_diff_subspace_v2"
    cfg_dict = asdict(cfg)
    intervene_diff = Intervention(cfg.intervention_func, layers_range)
    latent_diff_rev, logits_diff_rev = get_logits(correct_dataset, model, intervention=intervene_diff, **cfg_dict)
    latent_diff_rev = latent_diff_rev.float()
    logits_diff_rev = logits_diff_rev.float()
    stats = plot_logit_lens_latents(logits_diff, correct_dataset, **cfg_dict, title=f"reject {rev_lens_scale=}", cfg=cfg, only_compute_state = False)
    best_rev[rev_lens_scale] = stats
    latent_heatmap(latent_diff_rev, **cfg_dict, title=f"reject {rev_lens_scale=} scale_coeff {steer_scale_coeff}", cfg=cfg, bin_range = (0,10))

# %%
