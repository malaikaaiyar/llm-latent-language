# %%

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_lens import HookedTransformer, utils
#!pip install git+https://github.com/callummcdougall/eindex.git
import eindex
import dq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from dq_utils import plot_ci
from torch import Tensor
from jaxtyping import Float, Int
from typing import List
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm
import numpy as np
# %%
@torch.no_grad
def get_logits(dataset, model, intervention=None, **kwargs):
    """
    Compute the logits for a given dataset using a language model.

    Args:
        dataset (list): The dataset to compute logits for.
        intervention (object, optional): An intervention object for forward hooks. Defaults to None.
        tuned_lens (object, optional): An object for tuning the latents. Defaults to None.
        model (object, optional): The language model. Defaults to model.
        tokenizer (object, optional): The tokenizer. Defaults to tokenizer.

    Returns:
        torch.Tensor: The computed logits for the dataset.
    """
    
    device = next(model.parameters()).device
    tokenizer = model.tokenizer
    tuned_lens = model.tuned_lens
    use_tuned_lens = kwargs.get('use_tuned_lens', False)
    def get_latents(tokens, datapoint):
        """
        Get the latents for a given input sequence.

        Args:
            tokens (torch.Tensor): The input sequence tokens.
            datapoint (object, optional): The datapoint object. Defaults to None.

        Returns:
            torch.Tensor: The latents for the input sequence.
        """
        all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
        
        if intervention is None:
            fwd_hooks = []
        else:
            fwd_hooks = intervention.fwd_hooks(model, **datapoint, **kwargs)
                
        with model.hooks(fwd_hooks=fwd_hooks):
            output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
            
        latents = [act[:, -1, :] for act in cache.values()]
        latents = torch.stack(latents, dim=1)
        return latents.float()
    
    def unemb(latents):
        """
        Compute the logits for a given set of latents.
        """
        latents_ln = model.ln_final(latents)
        logits = latents_ln @ model.unembed.W_U + model.unembed.b_U
        return logits 
    
    all_logits = []
        
    with torch.no_grad():
        for idx, d in tqdm(enumerate(dataset), total=len(dataset)):
            
            tokens = tokenizer.encode(d['prompt'], return_tensors="pt").to(device)
            latents = get_latents(tokens, d)
            if use_tuned_lens:
                logits = torch.stack([tuned_lens(latents[:, i], i) for i in range(model.cfg.n_layers)], dim=1)
            else:
                logits = unemb(latents)
            
            all_logits.append(logits)
        
    all_logits = torch.stack(all_logits)
    return latents.squeeze(), all_logits.squeeze()

# @torch.no_grad
# def logit_lens(dataset, model, tuned_lens=None):
#     """
#     Measure language probabilities for a given dataset.

#     Args:
#         dataset (iterable): The dataset to measure language probabilities on.
#         steer (str, optional): The steering method. Defaults to None.
#             unembed: Use the unembeeded vectors of the latent ids
#             embed: Use the embedded vectors of the latent ids
#             both: Use both the unembeeded and embedded vectors of the latent ids
#         model (Model, optional): The language model. Defaults to model.
#         tokenizer (Tokenizer, optional): The tokenizer. Defaults to tokenizer.
#         device (str, optional): The device to run the model on. Defaults to device.

#     Returns:
#         tuple: Logits from each layer. You work out what to do with it.
#     """
#     tokenizer = model.tokenizer
#     device = next(model.parameters()).device
    
#     def get_latents(tokens, model):
#         all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
#         # if latent_ids is None:
#         #     output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
#         # else:    
#         #     subspace = model.unembed.W_U.T[latent_ids]
        
#         with model.hooks(fwd_hooks=[]):
#             output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
            
#         latents = [act[:, -1, :] for act in cache.values()]
#         #latents = [cache[f'blocks.{i}.hook_resid_post'][:, -1, :] for i in range(model.cfg.n_layers)] 
#         latents = torch.stack(latents, dim=1)
#         return latents #(batch=1, num_layers, d_model)
    
#     def unemb(latents, model):
#         latents_ln = model.ln_final(latents)
#         logits = latents_ln @ model.unembed.W_U + model.unembed.b_U
#         return logits 

#     all_logits = []
        
#     with torch.no_grad():
#         for idx, d in tqdm(enumerate(dataset), total=len(dataset)):
            
#             latent_ids = d['latent_ids']
#             out_ids = d['out_ids']
            
#             tokens = tokenizer.encode(d['prompt'], return_tensors="pt").to(device)
            
#             latents = get_latents(tokens, model)
#             if tuned_lens is not None:
#                 logits = torch.stack([tuned_lens(latents[:,i],i) for i in range(model.cfg.n_layers)], dim=1)
#             else:
#                 logits = unemb(latents, model) #(batch=1, num_layers, vocab_size)
#             #last = logits.softmax(dim=-1).detach().cpu().squeeze()
#             all_logits.append(logits)
#     all_logits = torch.stack(all_logits)
#     return all_logits.float().squeeze()

def compute_layer_probs(logprobs: Float[Tensor, "num_data num_layer vocab"],
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

from matplotlib.offsetbox import AnchoredText

def plotter(logprobs_list, label_list, **kwargs):
    
    out_path = kwargs.get('out_path', None)
    title = kwargs.get('title', None)
    figsize = kwargs.get('figsize', (6, 12))
    stats = kwargs.get('stats', None)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    if stats is not None:
        stats_text = "\n".join(f"{key}: {value:.03f}" for key, value in stats.items())
        # fig.text(0.75, 0.5, stats_text, 
        #          bbox=dict(facecolor='white', alpha=0.5), 
        #          fontsize=10, 
        #          verticalalignment='upper left')
        
        anchored_text = AnchoredText(stats_text, loc='center left', borderpad=1.0, frameon=True)
        anchored_text.patch.set_facecolor('white')
        anchored_text.patch.set_alpha(0.5)
        anchored_text.prop.set_size(12)  # Set the font size to 8
        ax2.add_artist(anchored_text)  # Add to the first subplot (ax1) or choose another as needed
    
    for logprobs, label in zip(logprobs_list, label_list):
        plot_ci(logprobs, ax1, dim=0, label=label)
        plot_ci(torch.exp(logprobs), ax2, dim=0, label=label)
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


def plot_logit_lens_latents(logits : Float[Tensor, "num_data num_layer vocab"], 
                            dataset,
                            **kwargs):
    
    cfg = kwargs.get('cfg', None)
    dest_lang = cfg.dest_lang
    latent_lang = cfg.latent_lang
    
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_list = []
    
    id_names = ['latent_ids', 'out_ids', 'alt_latent_ids', 'alt_out_ids']
    for name in id_names:
        ids = ([d[name] for d in dataset])
        logprobs_list.append(compute_layer_probs(logprobs, ids))
    correct_logprobs = logprobs_list[1][:, -1].mean().item()
    alt_logprobs = logprobs_list[3][:, -1].mean().item()
    correct_probs = torch.exp(logprobs_list[1][:, -1]).mean().item()
    alt_probs = torch.exp(logprobs_list[3][:, -1]).mean().item()
    stats = {'lp_out' : correct_logprobs, 
             'lp_alt' : alt_logprobs,
             'p_out' : correct_probs,
                'p_alt' : alt_probs,
             'lp_diff' : alt_logprobs - correct_logprobs,
             'p_ratio' : alt_probs / correct_probs}
    plotter(logprobs_list, [latent_lang, dest_lang, latent_lang + "_alt", dest_lang + "_alt"], stats = stats, **kwargs)
    return stats

def latent_heatmap(data, num_bins=250, bin_range=(0, 5), **kwargs):
    """
    Plot a heatmap of histograms for each row of the input data with a fixed range and specified bins.
    Uses logarithmic coloring for both imshow and the colorbar.

    Args:
    - data: A tensor of shape (num_rows, num_values).
    - num_bins: Number of bins to use for the histograms.
    - bin_range: The range for the histogram bins.

    Returns:
    - None
    """
    title = kwargs.get('title', None)
    
    data_np = data.numpy()
    
    # Calculate histograms for each row
    histograms = []
    for row in data_np:
        hist, bins = np.histogram(row, bins=num_bins, range=bin_range, density=True)
        histograms.append(hist)

    # Convert the list of histograms to a 2D array
    histograms_np = np.maximum(1.0 * np.array(histograms).T, 1e-3)[::-1]

    # Plot the heatmap using Matplotlib with logarithmic coloring
    plt.figure(figsize=(12, 8))
    plt.imshow(histograms_np, aspect='auto', cmap='viridis', norm=LogNorm(), extent=[0, data.shape[0], bin_range[0],bin_range[1]])
    cbar = plt.colorbar(label='Density', norm=LogNorm())
    cbar.set_label('Log Density')
    plt.title(title)
    plt.ylabel('Histogram Bin')
    plt.xlabel('Layer Number')
    plt.show()



# %%
# @torch.no_grad
# def batch_logit_lens(nn_model, prompts, only_last_token=True):
#     """
#     Compute the logits for each layer of a neural network model given a set of prompts.

#     Args:
#         nn_model (torch.nn.Module): The neural network model.
#         prompts (list[str]): The list of prompts.
#         only_last_token (bool, optional): Whether to consider only the last token of each prompt. 
#             Defaults to True.

#     Returns:
#         torch.Tensor: The logits per layer of the model.

#     """
#     model.eval()
#     tok_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
#     # Todo?: This is a hacky way to get the last token index for each prompt
#     last_token_index = tok_prompts.attention_mask.cumsum(1).argmax(-1)
    
#     output, cache = model.run_with_cache(prompts) #Expensive!
    
#     hidden_l = []
    
#     for i in range(model.cfg.n_layers):
#         layer_cache = cache[f'blocks.{i}.hook_resid_post']  # (batch, seq, d_model)
#         if only_last_token:
#             layer_cache = eindex(layer_cache, last_token_index, "i [i] j") # (batch, d_model)
#         hidden_l.append(layer_cache) # (batch, seq?, d_model)
            
#     hidden = torch.stack(hidden_l, dim=1)  # (batch, num_layers, seq?, d_model)
#     rms_out_ln = model.ln_final(hidden) # (batch, num_layers, seq?, d_model)
#     logits_per_layer = model.unembed(rms_out_ln) # (batch, num_layers, seq?, vocab_size)
    
#     return logits_per_layer
# %%




# %%
# prompt_zh_to_fr = "中文:花 français: fleur 中文:山 français: montagne 中文:月 français: lune 中文:水 français: eau"

# df = pd.read_csv('data/test/single_tok_lang6.csv')

# # %%
# def find_common_prefix(strings):
#     if not strings:
#         return ""
    
#     prefix = strings[0]
#     for string in strings[1:]:
#         while not string.startswith(prefix):
#             prefix = prefix[:-1]
#             if not prefix:
#                 return ""
    
#     return prefix


# @torch.no_grad
# def logit_lens_fast(nn_model, prompts):
#     """
#     Get the probabilities of the next token for the last token of each prompt at each layer using the logit lens.

#     Args:
#         nn_model: NNSight LanguageModel object
#         prompts: List of prompts or a single prompt
#         qk_caching: Perform a forward pass on the common prefix of the prompts to cache the query and key tensors.

#     Returns:
#         A tensor of shape (num_layers, num_prompts, vocab_size) containing the probabilities
#         of the next token for each prompt at each layer. Tensor is on the CPU.
#     """
    
#     prefix = find_common_prefix(prompts)
    
#     model.eval()
#     tok_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
#     # Todo?: This is a hacky way to get the last token index for each prompt
#     last_token_index = tok_prompts.attention_mask.cumsum(1).argmax(-1)
    
#     output, cache = model.run_with_cache(prompts) #Expensive!
    
#     hidden_l = []
#     for i in range(model.cfg.n_layers):
#         hidden_l.append(eindex(cache[f'blocks.{i}.hook_resid_post'], last_token_index, "i [i] j"))
    
#     hiddens = torch.stack(hidden_l, dim=0)  # (num_layers, num_prompts, d_model) 
#     rms_out_ln = model.ln_final(hiddens) # (num_layers, num_prompts, d_model)
#     logits_per_layer = model.unembed(rms_out_ln) # (num_layers, num_prompts, vocab_size)
#     probs_per_layer = logits.softmax(-1) # (num_layers, num_prompts, vocab_size)
    
    
#     assert torch.allclose(
#         logits_per_layer[-1],
#         eindex(output, last_token_index, "i [i] j")
#     )
#     return probs_per_layer

# # %%
# # Common prefix
# common_prefix = "The cat sat on the"

# # Unique suffixes
# suffixes = ["mat.", "rug.", "floor."]

# # Tokenize the common prefix
# tokens_prefix = tokenizer(common_prefix, return_tensors="pt")
# with torch.no_grad():
#     # Forward pass for the common prefix to obtain the last hidden state
#     outputs_prefix = model(**tokens_prefix, output_hidden_states=True)
#     hidden_states_prefix = outputs_prefix.hidden_states[-1][:, -1, :]

# # Process each suffix
# for suffix in suffixes:
#     # Tokenize suffix
#     tokens_suffix = tokenizer(suffix, return_tensors="pt")
#     # Concatenate the last hidden state of the prefix with the input ids of the suffix
#     # Adjust depending on whether you need to include the prefix's last token for context
#     inputs_with_state = {"input_ids": tokens_suffix["input_ids"], "past_key_values": outputs_prefix.past_key_values}
    
#     with torch.no_grad():
#         # Forward pass for the suffix, reusing the hidden state
#         outputs_suffix = model(**inputs_with_state)
    
#     # Convert logits to probabilities
#     probabilities = torch.softmax(outputs_suffix.logits[:, -1, :], dim=-1)
    
#     # Here you can extract and process the predictions as needed
#     print(f"Processed suffix '{suffix}'")