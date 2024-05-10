# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
from dataclasses import dataclass
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
from typing import Tuple, List, Optional, Dict, Callable
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookPoint
from beartype import beartype

# fix random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
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

def get_logits(dataset, steer=None, model=model, tokenizer=tokenizer, device=device):
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
        last_tblock = resid[:, -1]
        # subspace = W_U.T[latent_tok_ids]
        resid_subspace = proj(last_tblock.float(), subspace.float())
        #resid_alt = proj(last_tblock.float(), subspace_alt.float())
        
        norm_resid_subspace = torch.linalg.norm(resid_subspace)
        #norm_resid_alt=  torch.linalg.norm(resid_alt)
        
        #print(f"Norm of component to remove {norm_resid_subspace:.3f} Norm to restore {norm_resid_alt:.3f}")
        fv = subspace_alt.mean(dim=0)
        fv = fv / torch.linalg.norm(fv)
        
        resid[:, -1] = last_tblock - resid_subspace + norm_resid_subspace * fv
        return resid
    
    
    def get_latents(tokens, model, steer = None, datapoint = None):
        all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
        # if latent_ids is None:
        #     output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
        # else:    
        #     subspace = model.unembed.W_U.T[latent_ids]
        
            
        if steer is None:
            hook_all_resid = []
        else:
            # hook into the residual stream after each layer
            
            latent_token_ids = datapoint['latent_token_ids']
            alt_latent_token_ids = datapoint['alt_latent_token_ids']
            
            
            if steer == 'unembed':
                subspace = model.unembed.W_U.T[latent_token_ids]
                temp_hook_fn = lambda resid, hook: resid_stream_reject_subspace(resid, hook, subspace)
                hook_all_resid = [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in range(model.cfg.n_layers)]
            elif steer == "alt":
                subspace = model.unembed.W_U.T[latent_token_ids]
                subspace_alt = model.unembed.W_U.T[alt_latent_token_ids]
                temp_hook_fn = lambda resid, hook: resid_stream_move_subspace(resid, hook, subspace, subspace_alt)
                hook_all_resid = [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in range(15, 25)]
            
            else:
                raise ValueError("Invalid steering method")      
            
        with model.hooks(fwd_hooks=hook_all_resid):
            output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
            
        latents = [act[:, -1, :] for act in cache.values()]
        #latents = [cache[f'blocks.{i}.hook_resid_post'][:, -1, :] for i in range(model.cfg.n_layers)] 
        latents = torch.stack(latents, dim=1)
        return latents
    
    def unemb(latents, model):
        latents_ln = model.ln_final(latents)
        logits = latents_ln @ model.unembed.W_U + model.unembed.b_U
        return logits 
        
    all_logits = []
        
    with torch.no_grad():
        for idx, d in tqdm(enumerate(dataset), total=len(dataset)):
            
            latent_token_ids = d['latent_token_ids']
            out_token_ids = d['out_token_ids']
            
            tokens = tokenizer.encode(d['prompt'], return_tensors="pt").to(device)
            
            latents = get_latents(tokens, model, steer = steer, datapoint = d)
            logits = unemb(latents, model) #(batch=1, num_layers, vocab_size)
            #last = logits.softmax(dim=-1).detach().cpu().squeeze()
            all_logits.append(logits)
            
            
            # latent_token_probs += [last[:, latent_token_ids].sum(dim=-1)]
            # out_token_probs += [last[:, out_token_ids].sum(dim=-1)]
            # entropy += [compute_entropy(last)]
            #latents_all += [latents[:, -1, :].float().detach().cpu().clone()]
            # latents_normalized = latents[:, -1, :].float()
            # latents_normalized = latents_normalized / (((latents_normalized**2).mean(dim=-1, keepdim=True))**0.5)
            # latents_normalized /= (latents_normalized.norm(dim=-1, keepdim=True))
            # norm = ((U_normalized @ latents_normalized.T)**2).mean(dim=0)**0.5
            # energy += [norm/avgUU]
        

    # latent_token_probs = torch.stack(latent_token_probs)
    # out_token_probs = torch.stack(out_token_probs)
    # entropy = torch.stack(entropy)
    all_logits = torch.stack(all_logits)
    return all_logits
#energy = torch.stack(energy)
#latents = torch.stack(latents_all)
# %%

def compute_layer_probs(probs: Float[Tensor, "num_vocab"],
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
    layer_probs = []
    for i, tok_id in enumerate(token_ids):
        layer_prob = probs[i, :, tok_id].sum(dim=-1) #(num_layers)
        layer_probs.append(layer_prob.detach().cpu())
    return torch.stack(layer_probs)
# %%
#latent_token_probs, out_token_probs, entropy = measure_lang_probs(dataset)
# EXPENSIVE!
from dq_utils import plot_ci as plot_ci_dq

def plotter(prob_list, label_list, out_path=None, title=None):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 12))
    
    for prob, label in zip(prob_list, label_list):
        plot_ci_dq(torch.log2(prob), ax1, dim=0, label=label)
        plot_ci_dq(prob, ax2, dim=0, label=label)
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
def run(dataset, id_list, steer=None):
    logits = get_logits(dataset, steer).squeeze()
    probs = torch.softmax(logits, dim=-1)
    prob_list = []
    for ids in id_list:
        prob_list.append(compute_layer_probs(probs, ids))
    return prob_list

latent_token_ids = [d['latent_token_ids'] for d in dataset]
out_token_ids = [d['out_token_ids'] for d in dataset]
alt_latent_token_ids = [d['alt_latent_token_ids'] for d in dataset]
alt_out_token_ids = [d['alt_out_token_ids'] for d in dataset]

all_ids = [latent_token_ids, out_token_ids, alt_latent_token_ids, alt_out_token_ids]

# latent_token_probs, out_token_probs = run(dataset, [latent_token_ids, out_token_ids])
# latent_steer_token_probs, out_steer_token_probs = run(dataset, [latent_token_ids, out_token_ids], steer = 'unembed')

# %%
lat_control, out_control, alt_lat_control, alt_out_control = run(dataset, all_ids)
plotter([lat_control, out_control, alt_lat_control, alt_out_control], ["en", "zh", "en_alt", "zh_alt"], title= "no intervention")
# %%

# %%


# %%
plotter([latent_token_probs, out_token_probs, latent_steer_token_probs, out_steer_token_probs], ["en", "zh", "en_ablate", "zh_ablate"])





# %%
