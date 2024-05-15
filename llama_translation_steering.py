# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
from dataclasses import dataclass, field
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
import gen_data
from utils import plot_ci_plus_heatmap
from tuned_lens_wrap import load_tuned_lens
from dq_utils import proj, entropy, plot_ci
from logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap
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
    latent_lang: str = 'en'
    model_name: str = 'meta-llama/Llama-2-7b-hf'
    single_token_only: bool = False
    multi_token_only: bool = False
    token_add_prefixes : bool = True
    token_add_spaces : bool = True
    token_add_leading_byte : bool = True
    dataset_only_keep_correct : bool = True
    out_dir: str = './visuals'
    hf_token: str = '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'
    dataset_path: str = "./data/langs/"
    debug: bool = True
    num_multi_shot : int = 5
    use_tuned_lens: bool = True

cfg = Config()

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
tokenizer_vocab = tokenizer.get_vocab()

df_src = pd.read_csv(os.path.join(cfg.dataset_path, cfg.src_lang, 'clean.csv')).reindex()
df_dest = pd.read_csv(os.path.join(cfg.dataset_path, cfg.dest_lang, 'clean.csv')).reindex()
df_raw_data = gen_data.merge_datasets(df_src, df_dest, tokenizer_vocab, cfg)
dataset = gen_data.gen_translation_task(df_raw_data, tokenizer_vocab, cfg, return_tensors = "pt")


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#hf_model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, load_in_8bit=True)
# %%
# Function to load model
LOAD_MODEL = False
if LOAD_MODEL:
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype=torch.float16,
                                                            low_cpu_mem_usage=True)
            
    tuned_lens = load_tuned_lens(model)    
    model.tuned_lens = tuned_lens
# %%
if cfg.dataset_only_keep_correct:
    dataset = gen_data.filter_correct(dataset, model, cfg)

#energy = torch.stack(energy)
#latents = torch.stack(latents_all)
# %%


class Intervention:
    def __init__(self, func: Callable[..., Tensor], layers: List[int], **kwargs2):
        self.func = func
        self.layers = layers
        self.args = kwargs2
        
    def apply(self, resid: Tensor, hook: Any, model, **kwargs) -> Tensor:
        return self.func(resid, hook, model, **kwargs)

    def fwd_hooks(self, model: Any, datapoint : Dict[str, Any]) -> List:
        temp_hook_fn = lambda resid, hook: self.apply(resid, hook, model, **datapoint, **self.args)
        return [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in self.layers]

@beartype
def hook_reject_subspace(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids : Int[Tensor, "num_latent_tokens"] = None,
) -> Float[Tensor, "batch seq dmodel"]:
    # modify attn_pattern (can be inplace)
    subspace = model.unembed.W_U.T[latent_ids]
    last_tblock = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    last_tblock = last_tblock - proj(last_tblock.float(), subspace.float())
    resid[:, -1] = last_tblock
    return resid

@beartype
def hook_move_subspace(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    subspace = model.unembed.W_U.T[latent_ids]
    subspace_alt = model.unembed.W_U.T[alt_latent_ids]
    
    v = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    proj_correct = proj(v.float(), subspace.float()).half()
    #resid_alt = proj(last_tblock.float(), subspace_alt.float())
    proj_counter = proj(v.float(), subspace_alt.float()).half()
    resid[:, -1] = v - proj_correct + proj_counter
    return resid

@beartype
def hook_move_subspace2(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    scale_coeff : Float = 1,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    subspace = model.unembed.W_U.T[latent_ids]
    subspace_alt = model.unembed.W_U.T[alt_latent_ids]
    
    v = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    proj_correct = proj(v.float(), subspace.float()).half()
    #resid_alt = proj(last_tblock.float(), subspace_alt.float())
    proj_counter = proj(proj_correct.float(), subspace_alt.float()).half()
    resid[:, -1] = v - proj_correct + scale_coeff * proj_counter
    return resid


intervene_reject = Intervention(hook_reject_subspace, range(32))
intervene_move = Intervention(hook_move_subspace, range(32))
intervene_proj2 = Intervention(hook_move_subspace2, range(32))

@beartype
def hook_reject_tok(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids : Int[Tensor, "num_latent_tokens"] = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    # modify attn_pattern (can be inplace)
    subspace = model.unembed.W_U.T[latent_ids]
    last_tblock = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    last_tblock = last_tblock - proj(last_tblock.float(), subspace.float())
    resid[:, -1] = last_tblock
    return resid


# %%
#latent_probs, out_probs, entropy = measure_lang_probs(dataset)
# EXPENSIVE!
from logit_lens import get_logits
cfg = Config(use_tuned_lens = False)
#latents, logits = get_logits(dataset, model, intervention=None, cfg=cfg)
#latents_int, logits_int = get_logits(dataset, model, intervention=intervene_reject, cfg=cfg)
latents_bad, logits_bad = get_logits(dataset, model, intervention=intervene_proj2, cfg=cfg)
# %%
def analysis(latents, logits, **kwargs):
    stats = plot_logit_lens_latents(logits_bad, dataset, cfg=cfg, **kwargs)
    latent_heatmap(latents.cpu().abs(), **kwargs)



# %%
# end = 29
# for start in range(13,end):
#     layers = range(start, end)
#     intervene_move = Intervention(hook_move_subspace, layers)
#     logits = get_logits(dataset, model, intervention=intervene_move, cfg=cfg)
#     plot_logit_lens_latents(logits, dataset, cfg=cfg, title = f"move {start}-{end}")
# %%




# Example usage
# %%
