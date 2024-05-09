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

cfg = Config()

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
tokenizer_vocab = tokenizer.get_vocab()

df_src = pd.read_csv(os.path.join(cfg.dataset_path, cfg.src_lang, 'clean.csv')).reindex()
df_dest = pd.read_csv(os.path.join(cfg.dataset_path, cfg.dest_lang, 'clean.csv')).reindex()
df_raw_data = gen_data.merge_datasets(df_src, df_dest, tokenizer_vocab, cfg)
dataset = gen_data.gen_translation_task(df_raw_data, tokenizer_vocab, cfg)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
#hf_model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, load_in_8bit=True)
# %%
model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                        device='cuda:0', 
                                                        low_cpu_mem_usage=True,
                                                        dtype=torch.float16)

# model = "meta-llama/Llama-2-7b-hf"
# tokenizer = AutoTokenizer.from_pretrained(model, use_auth_token=hf_token)
#         self.model = AutoModelForCausalLM.from_pretrained(model, use_auth_token=hf_token,
#                                                           device_map=device_map,
#                                                           load_in_8bit=load_in_8bit)
# %%


# %%
id2voc = {id:voc for voc, id in tokenizer.get_vocab().items()}
def get_tokens(token_ids, id2voc=id2voc):
    return [id2voc[tokid] for tokid in token_ids]

def compute_entropy(probas):
    probas = probas[probas>0]
    return (-probas*torch.log2(probas)).sum(dim=-1)


# %%

from torch import Tensor
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookPoint
from beartype import beartype

@beartype
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

def measure_lang_probs(dataset, steer=None, model=model, tokenizer=tokenizer, device=device):
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
        tuple: A tuple containing the latent token probabilities, out token probabilities, and entropy.
    """
    
    
    
    in_token_probs = []
    latent_token_probs = []
    out_token_probs = []
    entropy = []
    #energy = []
    latents_all = []
    
    @beartype
    def hook_function(
        tblock_output: Float[Tensor, "batch seq dmodel"],
        hook: HookPoint,
        subspace: Float[Tensor, "num_vec dmodel"],
        # latent_tok_ids: Int[Tensor, "num_latent_tokens"],
    ) -> Float[Tensor, "batch seq dmodel"]:
        # modify attn_pattern (can be inplace)
        
        last_tblock = tblock_output[:, -1]
        # subspace = W_U.T[latent_tok_ids]
        last_tblock = rejection(last_tblock.float(), subspace.float())
        tblock_output[:, -1] = last_tblock
        return tblock_output
    
    def get_latents(tokens, model, subspace=None):
        all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
        # if latent_ids is None:
        #     output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
        # else:    
        #     subspace = model.unembed.W_U.T[latent_ids]
        if subspace is not None:
            temp_hook_fn = lambda tblock_output, hook: hook_function(tblock_output, hook, subspace)
            hook_all_resid = [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in range(model.cfg.n_layers)]
        else:
            hook_all_resid = []
            
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
        
    with torch.no_grad():
        for idx, d in tqdm(enumerate(dataset)):
            
            latent_token_ids = eval(d['latent_token_id']) if type(d['latent_token_id']) == str else d['latent_token_id']
            out_token_ids = eval(d['out_token_id']) if type(d['out_token_id']) == str else d['out_token_id']
            latent_token_ids = torch.tensor(latent_token_ids)
            out_token_ids = torch.tensor(out_token_ids)
            
            
            tokens = tokenizer.encode(d['prompt'], return_tensors="pt").to(device)
            
            latent_unembed = model.unembed.W_U.T[latent_token_ids]
            latent_embed = model.embed.W_E[latent_token_ids]
            
            if steer is None:
                subspace = None
            elif steer == 'unembed':
                subspace = latent_unembed
            elif steer == 'embed':
                subspace = latent_embed
            elif steer == 'both':
                subspace = torch.cat([latent_unembed, latent_embed], dim=0)
            else:
                raise ValueError("Invalid steering method")
            
            
            latents = get_latents(tokens, model, subspace)
            logits = unemb(latents, model) #(batch=1, num_layers, vocab_size)
            last = logits.softmax(dim=-1).detach().cpu().squeeze()
            
            
            
            latent_token_probs += [last[:, latent_token_ids].sum(dim=-1)]
            out_token_probs += [last[:, out_token_ids].sum(dim=-1)]
            entropy += [compute_entropy(last)]
            #latents_all += [latents[:, -1, :].float().detach().cpu().clone()]
            # latents_normalized = latents[:, -1, :].float()
            # latents_normalized = latents_normalized / (((latents_normalized**2).mean(dim=-1, keepdim=True))**0.5)
            # latents_normalized /= (latents_normalized.norm(dim=-1, keepdim=True))
            # norm = ((U_normalized @ latents_normalized.T)**2).mean(dim=0)**0.5
            # energy += [norm/avgUU]
        

    latent_token_probs = torch.stack(latent_token_probs)
    out_token_probs = torch.stack(out_token_probs)
    entropy = torch.stack(entropy)
    
    return latent_token_probs, out_token_probs, entropy
#energy = torch.stack(energy)
#latents = torch.stack(latents_all)
# %%
latent_token_probs, out_token_probs, entropy = measure_lang_probs(dataset)
# %%
latent_steer_token_probs, out_steer_token_probs, entropy_steer = measure_lang_probs(dataset, steer='unembed')
# %%
latent_embed_steer_token_probs, out_embed_steer_token_probs, entropy_embed_steer = measure_lang_probs(dataset, steer='embed')
# %%
latent_both_steer_token_probs, out_both_steer_token_probs, entropy_both_steer = measure_lang_probs(dataset, steer='both')
# %%
from dq_utils import plot_ci as plot_ci_dq

fig, ax = plt.subplots()
plot_ci_dq(latent_token_probs, ax, dim=0, label='en', color='blue')
plot_ci_dq(out_token_probs, ax, dim=0, label='zh', color='red')
#plot_ci_dq(latent_steer_token_probs, ax, dim=0, label='en_WU', color='green')
#plot_ci_dq(out_steer_token_probs, ax, dim=0, label='zh_WU', color='purple')
plot_ci_dq(latent_embed_steer_token_probs, ax, dim=0, label='en_WE', color='orange')
plot_ci_dq(out_embed_steer_token_probs, ax, dim=0, label='zh_WE', color='pink')
plt.legend()
plt.title(f"lang probs {model_name}")
plt.xlabel('Layer')
plt.ylabel('Probability')
plt.savefig(f"out/llama_steering_WE.svg", format='svg')
plt.show()


fig, ax = plt.subplots()
plot_ci_dq(latent_token_probs, ax, dim=0, label='en', color='blue')
plot_ci_dq(out_token_probs, ax, dim=0, label='zh', color='red')
plot_ci_dq(latent_steer_token_probs, ax, dim=0, label='en_WU', color='green')
plot_ci_dq(out_steer_token_probs, ax, dim=0, label='zh_WU', color='purple')
#plot_ci_dq(latent_embed_steer_token_probs, ax, dim=0, label='en_WE', color='orange')
#plot_ci_dq(out_embed_steer_token_probs, ax, dim=0, label='zh_WE', color='pink')
plt.legend()
plt.title(f"lang probs {model_name}")
plt.xlabel('Layer')
plt.ylabel('Probability')
plt.savefig(f"out/llama_steering_WU.svg", format='svg')
plt.show()

fig, ax = plt.subplots()
plot_ci_dq(latent_token_probs, ax, dim=0, label='en', color='blue')
plot_ci_dq(out_token_probs, ax, dim=0, label='zh', color='red')
plot_ci_dq(latent_both_steer_token_probs, ax, dim=0, label='en_both', color='green')
plot_ci_dq(out_both_steer_token_probs, ax, dim=0, label='zh_both', color='purple')
#plot_ci_dq(latent_embed_steer_token_probs, ax, dim=0, label='en_WE', color='orange')
#plot_ci_dq(out_embed_steer_token_probs, ax, dim=0, label='zh_WE', color='pink')
plt.legend()
plt.title(f"lang probs {model_name}")
plt.xlabel('Layer')
plt.ylabel('Probability')
plt.savefig(f"out/llama_steering_both.svg", format='svg')
plt.show()




# %%
