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
from scipy.stats import bootstrap
from utils import plot_ci, plot_ci_plus_heatmap
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer
import gen_data

from torch import Tensor
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Callable, Iterable
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookPoint
from beartype import beartype
from tuned_lens_wrap import load_tuned_lens
import einops
import torch.nn as nn
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
model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                        device=device, 
                                                        dtype = torch.float16,
                                                        low_cpu_mem_usage=True)
tuned_lens = load_tuned_lens(cfg.model_name)
#stuff to make Tuned Lens happy, expecting Huggingface model
# %%

#stuff to make Tuned Lens happy, expecting Huggingface model

# model = "meta-llama/Llama-2-7b-hf"
# tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=hf_token)
#         self.model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=hf_token,
#                                                           device_map=device_map,
#                                                           load_in_8bit=load_in_8bit)
# %%




# %%
# id2voc = {id:voc for voc, id in tokenizer.get_vocab().items()}
# def get_tokens(token_ids, id2voc=id2voc):
#     return [id2voc[tokid] for tokid in token_ids]

def compute_entropy(probas):
    probas = probas[probas>0]
    return (-probas*torch.log2(probas)).sum(dim=-1)


def proj(x : Float[Tensor, "... dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "... dmodel"]:
    # Computes the projection of x onto the subspace spanned by the columns of Y
    Y = Y.transpose(-2, -1) #(dmodel, numvec) #require column vectors
    # Solve the linear system (Y^T @ Y) @ c = Y^T @ x
    # c is the coefficients of the projection of x onto the subspace spanned by the columns of Y
    # so the projection of x onto the subspace spanned by the columns of Y is Y @ c
    if x.ndim == 1:
        x = x.unsqueeze(0)
    
    c = torch.linalg.solve(Y.transpose(-2, -1)  @ Y, (x @ Y).transpose(-2, -1))    
    proj_x = (Y @ c).transpose(-2, -1) 
    return proj_x.squeeze()


def rejection(x : Float[Tensor, "batch dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "batch dmodel"]:
    return x - proj(x, Y)
    
    
# %%
class BatchTunedLens(nn.Module):
    def __init__(self, tuned_lens):
        super(BatchTunedLens, self).__init__()
        self.unembed = tuned_lens.unembed
        
        num_lenses = len(tuned_lens.layer_translators)
        out_features, in_features = tuned_lens.layer_translators[0].weight.shape
        
        device = tuned_lens.layer_translators[0].weight.device  # Extract device from layer_lens.weight
        dtype = tuned_lens.layer_translators[0].weight.dtype    # Extract dtype from layer_lens.weight

        self.W_lens = nn.Parameter(torch.empty((num_lenses, out_features, in_features), device=device, dtype=dtype))
        self.b_lens = nn.Parameter(torch.empty((num_lenses, out_features), device=device, dtype=dtype))
        
        for i in range(num_lenses):
            self.W_lens[i].data.copy_(tuned_lens.layer_translators[i].weight)
            self.b_lens[i].data.copy_(tuned_lens.layer_translators[i].bias)
        
    def forward(self, h : Float[Tensor, "... num_layers dmodel"], skip_unembed = False):
        # Note that we add the translator output residually, in contrast to the formula
        # in the paper. By parametrizing it this way we ensure that weight decay
        # regularizes the transform toward the identity, not the zero transformation.
        # See https://github.com/AlignmentResearch/tuned-lens/blob/main/tuned_lens/nn/lenses.py#L311C32-L312C1
        # nn.Linear uses fused-multiply torch.addmm
        h_out = einops.einsum(h, self.W_lens, "... layers din, layers dout din -> ... layers dout") + self.b_lens
        new_h = h + h_out
        if skip_unembed:
            return new_h
        else:
            return self.unembed(new_h)

#tuned_lens.to(device)
#batched_tuned_lens = BatchTunedLens(tuned_lens).to(device)

# for i in range(32):
#     assert torch.equal(batched_tuned_lens.W_lens[i].data, tuned_lens.layer_translators[i].weight.data), "weight no match"
#     assert torch.equal(batched_tuned_lens.b_lens[i].data, tuned_lens.layer_translators[i].bias.data), "bias no match"
# print("weight/bias match!")

# resid = torch.randn(32, 4096, dtype = torch.float16, device=device)

# for i in range(32):
#     #y = ((resid @ batched_tuned_lens.W_lens[i].T) + batched_tuned_lens.b_lens[i])
#     #y2 = resid @ tuned_lens.layer_translators[0].weight.T + tuned_lens.layer_translators[0].bias
#     y2 = resid[i] + tuned_lens.layer_translators[i](resid[i])
#     y3 = batched_tuned_lens(resid, skip_unembed=True)[i]
#     assert torch.allclose(y2,y3, rtol=0.05, atol = 1e-4), f"failure in layer {i} {y2=} {y3=}"
# print("linear match!")
# # %%
# h = torch.randn(32, 4096).half().to(device)
# y = batched_tuned_lens(h)
# y2 = torch.stack([tuned_lens(h[i],i) for i in range(32)], dim=0)
# print(f"{y.shape=}, {y2.shape=}")
# torch.allclose(y,y2, rtol=0.05, atol = 1e-4)



# %%
# %%
def get_logits(dataset, cfg=None, model=model, tokenizer=tokenizer, device=device):
    """
    Measure language probabilities for a given dataset.

    Args:
        dataset (iterable): The dataset to measure language probabilities on.
        steer (str, optional): The steering method. Defaults to None.
            unembed: Use the unembeeded vectors of the latent ids
            embed: Use the embedded vectors of the latent ids
            both: Use both the unembeeded and embedded vectors of the latent ids
        model (Model, optional): The language model. Defaults to model.
        tokenizer (Tokenizer, optional): The tokenizer. Defaults to tokenizer.
        device (str, optional): The device to run the model on. Defaults to device.

    Returns:
        tuple: Logits from each layer. You work out what to do with it.
    """
    
    # in_token_probs = []
    # latent_token_probs = []
    # out_token_probs = []
    # entropy = []
    #energy = []
    latents_all = []
    
    @beartype
    def resid_stream_reject_subspace(
        resid: Float[Tensor, "batch seq dmodel"],
        hook: HookPoint,
        subspace: Float[Tensor, "num_vec dmodel"],
        # latent_tok_ids: Int[Tensor, "num_latent_tokens"],
    ) -> Float[Tensor, "batch seq dmodel"]:
        # modify attn_pattern (can be inplace)
        
        last_tblock = resid[:, -1]
        # subspace = W_U.T[latent_tok_ids]
        last_tblock = rejection(last_tblock.float(), subspace.float())
        resid[:, -1] = last_tblock
        return resid
    
    @beartype
    def resid_stream_move_subspace(
        resid: Float[Tensor, "batch seq dmodel"],
        hook: HookPoint,
        subspace: Float[Tensor, "num_vec dmodel"],
        subspace_alt: Float[Tensor, "num_vec2 dmodel"]
    ) -> Float[Tensor, "batch seq dmodel"]:
        v = resid[:, -1]
        # subspace = W_U.T[latent_tok_ids]
        proj_A_v = proj(v.float(), subspace.float())
        #resid_alt = proj(last_tblock.float(), subspace_alt.float())
        proj_B_proj_A_v = proj(proj_A_v, subspace_alt.float())
        #norm_resid_alt=  torch.linalg.norm(resid_alt)
        
        
        resid[:, -1] = v - proj_A_v + proj_B_proj_A_v
        return resid
    
    
    def get_latents(tokens, model, datapoint = None):
        all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
        # if latent_ids is None:
        #     output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
        # else:    
        #     subspace = model.unembed.W_U.T[latent_ids]
        
            
        if cfg is None or cfg.steer is None:
            hook_all_resid = []
        else:
            # hook into the residual stream after each layer
            steer = cfg.steer
            latent_ids = datapoint['latent_ids']
            alt_latent_ids = datapoint['alt_latent_ids']
            
            
            if steer == 'unembed':
                subspace = model.unembed.W_U.T[latent_ids]
                temp_hook_fn = lambda resid, hook: resid_stream_reject_subspace(resid, hook, subspace)
                hook_all_resid = [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in cfg.steer_layers]
            elif steer == "alt":
                subspace = model.unembed.W_U.T[latent_ids]
                subspace_alt = model.unembed.W_U.T[alt_latent_ids]
                temp_hook_fn = lambda resid, hook: resid_stream_move_subspace(resid, hook, subspace, subspace_alt)
                hook_all_resid = [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in cfg.steer_layers]
            
            else:
                raise ValueError("Invalid steering method")      
            
        with model.hooks(fwd_hooks=hook_all_resid):
            output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
            
        latents = [act[:, -1, :] for act in cache.values()]
        #latents = [cache[f'blocks.{i}.hook_resid_post'][:, -1, :] for i in range(model.cfg.n_layers)] 
        latents = torch.stack(latents, dim=1)
        return latents #(batch=1, num_layers, d_model)
    
    def unemb(latents, model):
        latents_ln = model.ln_final(latents)
        logits = latents_ln @ model.unembed.W_U + model.unembed.b_U
        return logits 
    
    
        
    all_logits = []
        
    with torch.no_grad():
        for idx, d in tqdm(enumerate(dataset), total=len(dataset)):
            
            latent_ids = d['latent_ids']
            out_ids = d['out_ids']
            
            tokens = tokenizer.encode(d['prompt'], return_tensors="pt").to(device)
            
            latents = get_latents(tokens, model, datapoint = d)
            if cfg.tuned_lens:
                logits = torch.stack([tuned_lens(latents[:,i],i) for i in range(model.cfg.n_layers)], dim=1)
            else:
                logits = unemb(latents, model) #(batch=1, num_layers, vocab_size)
            #last = logits.softmax(dim=-1).detach().cpu().squeeze()
            all_logits.append(logits)
            
            
            # latent_probs += [last[:, latent_ids].sum(dim=-1)]
            # out_probs += [last[:, out_ids].sum(dim=-1)]
            # entropy += [compute_entropy(last)]
            #latents_all += [latents[:, -1, :].float().detach().cpu().clone()]
            # latents_normalized = latents[:, -1, :].float()
            # latents_normalized = latents_normalized / (((latents_normalized**2).mean(dim=-1, keepdim=True))**0.5)
            # latents_normalized /= (latents_normalized.norm(dim=-1, keepdim=True))
            # norm = ((U_normalized @ latents_normalized.T)**2).mean(dim=0)**0.5
            # energy += [norm/avgUU]
        

    # latent_probs = torch.stack(latent_probs)
    # out_probs = torch.stack(out_probs)
    # entropy = torch.stack(entropy)
    all_logits = torch.stack(all_logits)
    return all_logits.float()
#energy = torch.stack(energy)
#latents = torch.stack(latents_all)
# %%


# %%
#latent_probs, out_probs, entropy = measure_lang_probs(dataset)
# EXPENSIVE!
from dq_utils import plot_ci as plot_ci_dq

def plotter(logprobs_list, label_list, out_path=None, title=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
    
    for logprobs, label in zip(logprobs_list, label_list):
        plot_ci_dq(logprobs, ax1, dim=0, label=label)
        plot_ci_dq(torch.exp(logprobs), ax2, dim=0, label=label)
    plt.legend()
    fig.suptitle(title)
    fig.tight_layout()  # Add this line to reduce the gap between subplots and title
    ax2.set_xlabel('Layer')
    ax1.set_ylabel('Log Probability')
    ax2.set_ylabel('Raw Probability')
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)  # Add minor gridlines to ax1
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)  # Add minor gridlines to ax2
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)  # Add minor gridlines to the whole figure
    if out_path is not None:
        plt.savefig(out_path, format='svg')
    plt.show()

# %%
def run(dataset, id_list, cfg=None):
    
    def compute_layer_probs(logprobs: Float[Tensor, "num_vocab"],
                        token_ids: List[Int[Tensor, "num_idx"]],
    ) -> Float[Tensor, "datapoints num_layers"]:
        """
        Compute the layer probabilities for each token ID.

        Args:
            probs (List[Float[Tensor, "num_vocab"]]): The probabilities for each token ID.
            token_ids (List[List[int]]): The token IDs for each datapoint.

        Returns:
            Float[Tensor, "datapoints num_layers"]: The layer probabilities for each datapoint.
        """
        layer_logprobs = []
        for i, tok_id in enumerate(token_ids):
            layer_logprob = torch.logsumexp(logprobs[i, :, tok_id], dim=-1) #(num_layers)
            layer_logprobs.append(layer_logprob.detach().cpu())
        return torch.stack(layer_logprobs)
    
    logits = get_logits(dataset, cfg=cfg).squeeze()
    logprobs = F.log_softmax(logits, dim=-1)
    logprob_list = []
    for ids in id_list:
        logprob_list.append(compute_layer_probs(logprobs, ids))
    return logprob_list

latent_ids = [d['latent_ids'] for d in dataset]
out_ids = [d['out_ids'] for d in dataset]
alt_latent_ids = [d['alt_latent_ids'] for d in dataset]
alt_out_ids = [d['alt_out_ids'] for d in dataset]

all_ids = [latent_ids, out_ids, alt_latent_ids, alt_out_ids]

# latent_probs, out_probs = run(dataset, [latent_ids, out_ids])
# latent_steer_probs, out_steer_probs = run(dataset, [latent_ids, out_ids], steer = 'unembed')

# %%


@dataclass
class SteerConfig:
    tuned_lens: bool = False
    steer: Optional[str] = None
    steer_layers: Iterable = field(default_factory=lambda: range(15,model.cfg.n_layers))


# Now try to create an instance with tuned_lens set to True
steercfg = SteerConfig()

# %%
labels = ["en", "zh", "en_alt", "zh_alt"]
no_intervention = run(dataset, all_ids, steercfg)
plotter(no_intervention, labels, title= f"{steercfg}")
# %%
labels = ["en", "zh", "en_alt", "zh_alt"]
no_intervention = run(dataset, all_ids, SteerConfig(tuned_lens=True))
plotter(no_intervention, labels, title= f"{steercfg}")


# %%
intervention = run(dataset, all_ids, SteerConfig(steer = "alt"))
plotter(intervention, labels, title= "fv?")

intervention = run(dataset, all_ids, SteerConfig(steer = "alt", tuned_lens=True, steer_layers=range(32)))
plotter(intervention, labels, title= "fv?")
# %%
for i in range(32):
    for j in range(i, 32):
        cfg = SteerConfig(steer = "alt", tuned_lens=True, steer_layers = range(i,j))
        intervention = run(dataset, all_ids, cfg=cfg)
        plotter(intervention, labels, title= f"{cfg}", out_path=f"out/fv_tuned_{i}_{j}.svg")
        
for i in range(32):
    for j in range(i, 32):
        cfg = SteerConfig(steer = "alt", tuned_lens=False, steer_layers = range(i,j))
        intervention = run(dataset, all_ids, cfg=cfg)
        plotter(intervention, labels, title= f"{cfg}", out_path=f"out/fv_{i}_{j}.svg")




# %%
