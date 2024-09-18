# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm import tqdm

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

# ==== Custom Libraries ====
import OLD_llama.gen_data as gen_data
from utils_plot import plot_ci_plus_heatmap
from tuned_lens_wrap import load_tuned_lens
from utils.misc import proj, entropy, plot_ci
from src.logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap
# %%
# fix random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_grad_enabled(False)
# %%

@dataclass
class Config:
    seed: int = 42
    src_lang: str = 'fr'
    dest_lang: str = 'zh'
    latent_lang: str = 'de'
    model_name: str = 'meta-llama/Llama-2-7b-hf'
    single_token_only: bool = False
    multi_token_only: bool = False
    token_add_prefixes : bool = True
    token_add_spaces : bool = True
    token_add_leading_byte : bool = True
    dataset_filter_correct : bool = True
    out_dir: str = './visuals'
    hf_token: str = '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'
    dataset_path: str = "./data/langs/"
    debug: bool = True
    num_multi_shot : int = 5
    use_tuned_lens: bool = True
    return_tensors: str = "pt"

cfg = Config()
cfg_kwargs = asdict(cfg)
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
tokenizer_vocab = tokenizer.get_vocab()

df_src = pd.read_csv(os.path.join(cfg.dataset_path, cfg.src_lang, 'clean.csv')).reindex()
df_dest = pd.read_csv(os.path.join(cfg.dataset_path, cfg.dest_lang, 'clean.csv')).reindex()
df_latent = pd.read_csv(os.path.join(cfg.dataset_path, cfg.latent_lang, 'clean.csv')).reindex()
df_raw_data = gen_data.merge_datasets(df_src, df_dest, df_latent, tokenizer_vocab, **cfg_kwargs)
# %%

dataset = gen_data.gen_translation_task(df_raw_data, tokenizer_vocab, **cfg_kwargs)


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#hf_model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, load_in_8bit=True)
# %%
# Function to load model
LOAD_MODEL = True
if LOAD_MODEL:
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype=torch.float16,
                                                            low_cpu_mem_usage=True)
            
    tuned_lens = load_tuned_lens(model)    
    model.tuned_lens = tuned_lens.float()
# %%
if cfg.dataset_filter_correct:
    dataset = gen_data.filter_correct(dataset, model)
    

#energy = torch.stack(energy)
#latents = torch.stack(latents_all)
# %%

from src.intervention import Intervention
import src.intervention as intervention 

intervene_reject = Intervention(intervention.hook_reject_subspace, range(32))
intervene_move = Intervention(intervention.hook_move_subspace, range(32))
intervene_proj2 = Intervention(intervention.hook_move_subspace2, range(32))



# %%
#latent_probs, out_probs, entropy = measure_lang_probs(dataset)
# EXPENSIVE!
from src.logit_lens import get_logits
cfg = Config(use_tuned_lens = False)
latents, logits = get_logits(dataset, model, intervention=None, cfg=cfg)
#latents_int, logits_int = get_logits(dataset, model, intervention=intervene_reject, cfg=cfg)
latents_bad, logits_bad = get_logits(dataset, model, intervention=intervene_reject, **cfg_kwargs)
latents_wrong, logits_wrong = get_logits(dataset, model, intervention=intervene_reject, interv_match_latent =False, **cfg_kwargs)
# %%
def analysis(latents, logits, **kwargs):
    stats = plot_logit_lens_latents(logits, dataset, cfg=cfg, **kwargs)
    latent_heatmap(latents.cpu().abs(), **kwargs)
analysis(latents_bad, logits_bad, **cfg_kwargs, title="Intervention En")
analysis(latents, logits, **cfg_kwargs, title="No intevention")
analysis(latents_wrong, logits_wrong, **cfg_kwargs, title="Intervention En alt")
# %%