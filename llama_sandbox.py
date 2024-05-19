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
#measure_performance(correct_dataset, model)

# %%
