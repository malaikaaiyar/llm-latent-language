# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
from dataclasses import dataclass, field
import numpy as np
from matplotlib import pyplot as plt
import torch
import sys
import os
#from llamawrapper import load_unemb_only, LlamaHelper
# from scipy.stats import bootstrap
# from utils import plot_ci, plot_ci_plus_heatmap
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

from torch import Tensor
from torch.utils.data import DataLoader
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Callable, Iterable
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookPoint
from beartype import beartype
from tuned_lens import TunedLens
import einops
import torch.nn as nn

import gen_data
from dq_utils import plot_ci as plot_ci_dq
from dq_utils import proj, plotter, measure_performance
from logit_lens import logit_lens, plot_logit_lens_latents
from tuned_lens_wrap import load_tuned_lens
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
    latent_lang: str = 'en'
    model_size: str = '7b'
    model_name: str = 'meta-llama/Llama-2-%s-hf' % model_size
    single_token_only: bool = False
    multi_token_only: bool = False
    out_dir: str = './visuals'
    hf_token: str = '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'
    dataset_path: str = "./data/langs/"
    debug: bool = True
    num_multi_shot : int = 5
    token_add_spaces: bool = True
    token_add_leading_byte: bool = False
    token_add_prefixes : bool = True
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cfg = Config()

# tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
# tokenizer_vocab = tokenizer.get_vocab()

device = cfg.device
model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                        device=device, 
                                                        dtype = torch.float16)
tuned_lens = load_tuned_lens(model)
tokenizer_vocab = model.tokenizer.get_vocab()
df_src = pd.read_csv(os.path.join(cfg.dataset_path, cfg.src_lang, 'clean.csv')).reindex()
df_dest = pd.read_csv(os.path.join(cfg.dataset_path, cfg.dest_lang, 'clean.csv')).reindex()
df_raw_data = gen_data.merge_datasets(df_src, df_dest, tokenizer_vocab, cfg)
raw_dataset = gen_data.gen_translation_task(df_raw_data, tokenizer_vocab, cfg, return_tensors = "pt")


#hf_model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, load_in_8bit=True)
# %%
# %%
dataset = gen_data.purge_dataset(raw_dataset, model, cfg)
# %%
#measure_performance(dataset, model)


logits = logit_lens(raw_dataset, model, tuned_lens=tuned_lens)
plot_logit_lens_latents(logits, raw_dataset)
# %%




# %%

    
# @beartype
# def resid_stream_move_subspace(
#     resid: Float[Tensor, "batch seq dmodel"],
#     hook: HookPoint,
#     subspace: Float[Tensor, "num_vec dmodel"],
#     subspace_alt: Float[Tensor, "num_vec2 dmodel"],
#     layer_scale: Float[Tensor, "1"],
#     proj_scale: Float[Tensor, "2"],
# ) -> Float[Tensor, "batch seq dmodel"]:
#     v = resid[:, -1]
#     # subspace = W_U.T[latent_tok_ids]
#     proj_A_v = proj(v.float(), subspace.float())
#     #resid_alt = proj(last_tblock.float(), subspace_alt.float())
#     proj_B_v = proj(v.float(), subspace_alt.float())
#     #norm_resid_alt=  torch.linalg.norm(resid_alt)
    
#     v = v + layer_scale * (proj_scale[0] * proj_A_v + proj_scale[1] * proj_B_v)
#     return resid


# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
# for d in dataloader:
#     tok_prompt = tokenizer.encode(d['prompt'], return_tensors="pt").to(device)
#     latent_ids = d['latent_ids']
#     alt_latent_ids = d['alt_latent_ids']
#     out_ids = d['out_ids']
#     alt_out_ids = d['alt_out_ids']
    
    
    
    
    # all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    # with model.hooks(fwd_hooks=[(all_post_resid[0], lambda resid, hook: resid_stream_move_subspace(resid, hook, model.unembed.W_U.T[datapoint['latent_ids']], model.unembed.W_U.T[datapoint['alt_latent_ids']], torch.tensor([1.0]), torch.tensor([1.0])))]):
    #     output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
    # break    

# %%

# %%



# %%
