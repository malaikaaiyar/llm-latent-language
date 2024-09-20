# %%
%load_ext autoreload
%autoreload 2
# %%
from tracemalloc import start
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
import OLD_llama.gen_data as gen_data
from utils_plot import plot_ci_plus_heatmap
from tuned_lens_wrap import load_tuned_lens
from reverse_tuned_lens import ReverseLens
from utils.misc import proj, entropy, plot_ci, is_chinese_char, measure_performance
from src.logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap, get_logits_batched
import src.intervention as intervention
from src.intervention import Intervention
from utils.config_argparse import parse_args
from llama_merge_csv import construct_dataset
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
    use_tuned_lens : bool = False
    interv_match_latent : bool = True
    interv_steer_coeff : float = 1.0
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
    cache_prefix : bool = True

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
    #cfg.use_reverse_lens = True
    
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
df_raw_data = construct_dataset(**cfg_dict)
dataset = gen_data.gen_batched_dataset(df_raw_data, model.tokenizer, **cfg_dict)
# %%
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
from src.logit_lens import get_logits_batched
suffix_toks = dataset['suffixes']
prefix_toks = dataset['prompt_tok']

latents, logits = get_logits_batched(prefix_toks, suffix_toks, model, **cfg_dict)
#with_cache_logits = model(rest_of_tokens, past_kv_cache=kv_cache)
# %%
# %%
