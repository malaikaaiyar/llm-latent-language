
import warnings
import matplotlib.pyplot as plt
import torch
from typing import List
from jaxtyping import Float, Int
from torch import Tensor
from tqdm import tqdm
import pandas as pd
from transformer_lens import HookedTransformerKeyValueCache
import os
import sys
import pickle
import pprint
from dataclasses import asdict
import numpy as np
import gen_data
import prefix

def printd(*args, **kwargs):
    # Check if '__DEBUG__' is in the global namespace and if it is set to True
    if globals().get('__DEBUG__', False):
        print("DEBUG:", end=" ")
        print(*args, **kwargs)

def calculate_iterations(start_lower, start_upper, end_lower, end_upper):
    if start_upper <= start_lower or end_upper <= end_lower:
        return 0  # No valid iterations if ranges are non-positive or improperly defined

    # Maximum valid start_layer is start_upper - 1
    # Minimum valid end_layer is start_layer + 1, which translates to start_lower + 1 for start_lower
    if end_upper <= start_lower + 1:
        return 0  # No valid end_layer values if end_upper is less than or equal to start_lower + 1

    # Applying the formula: Summing (end_upper - k - 1) for k from start_lower to start_upper - 1
    total_iterations = 0
    for k in range(start_lower, start_upper):
        if k + 1 < end_upper:  # Ensure that there is at least one valid end_layer
            total_iterations += (end_upper - (k + 1))

    return total_iterations


def str_dict(d):
    # Create a formatted string from dictionary entries
    items = [f"{k}: {f'{v:.4f}' if isinstance(v, float) else v}" for k, v in d.items()]
    # Join all items in a single line
    return ', '.join(items)

def write_log(layer_log2, cfg, info = {}):
    base_log_file_path = cfg.log_file.rsplit('.', 1)[0]  # Strip off the extension if provided

    # Ensure directory exists
    os.makedirs(os.path.dirname(base_log_file_path), exist_ok=True)

    with open(base_log_file_path + ".pkl", "wb") as pickle_file:
        pickle.dump(layer_log2, pickle_file)

    # pickle.dump(layer_log2, open(cfg.log_file + ".pkl", "wb"))

    log_legend = """
    Measuring 
    lp_out/p_out : logprobs/probs of correct answer
    lp_alt/p_alt logprobs/probs of alternate answer
    lp_diff/p_ratio: logprob_diff/probs ration of alt-correct or alt/correct
    """

    pp = pprint.PrettyPrinter(sort_dicts=False)
    # Save log_legend to the log file
    with open(base_log_file_path + ".log", "a") as f:
        f.write("Command: " + ' '.join(sys.argv) + "\n")
        f.write(pp.pformat(asdict(cfg)))
        f.write("\n==============\n")
        for key, val in info.items():
            f.write(f"{key}: {val}\n")
    print("Done!")


def proj(x : Float[Tensor, "... dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "... dmodel"]:
    # Computes the projection of x onto the subspace spanned by the columns of Y
    if Y.dim() == 1:
        Y = Y.unsqueeze(0)
    Y = Y.mT #(dmodel, numvec) #require column vectors
    # Solve the linear system (Y^T @ Y) @ c = Y^T @ x
    # c is the coefficients of the projection of x onto the subspace spanned by the columns of Y
    # so the projection of x onto the subspace spanned by the columns of Y is Y @ c
    if x.ndim == 1:
        x = x.unsqueeze(0)
    
    c = torch.linalg.solve(Y.mT  @ Y, (x @ Y).mT)    
    proj_x = (Y @ c).mT 
    return proj_x.squeeze()

def entropy(probas):
    probas = probas[probas>0]
    return (-probas*torch.log2(probas)).sum(dim=-1)


def rejection(x : Float[Tensor, "batch dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "batch dmodel"]:
    return x - proj(x, Y)
    


lang2name = {'en' : "English", 
            'zh' : "中文", 
            "fr" : "français", 
            "ru" : "pусский",
            "de" : "deutsch",
            "ko" : "한국어"}

def is_chinese_char(ch):
    """Check if a character is a Chinese character using a list of Unicode ranges and return range information.
    Now robust to invalid inputs."""
    try:
        c = ord(ch)
    except:
        warnings.warn("is_chinese_char recieved non-char input", category = RuntimeWarning)
        return False
    # List of tuples, each representing a range of Chinese character code points with labels
    unicode_ranges = [
        (0x4E00, 0x9FFF, 'Common'),
        (0x3400, 0x4DBF, 'Extension A'),
        (0x20000, 0x2A6DF, 'Extension B'),
        (0x2A700, 0x2B73F, 'Extension C'),
        (0x2B740, 0x2B81F, 'Extension D'),
        (0x2B820, 0x2CEAF, 'Extension E'),
        (0x2CEB0, 0x2EBEF, 'Extension F'),
        (0x30000, 0x3134F, 'Extension G'),
        (0x31350, 0x323AF, 'Extension H'),
        (0xF900, 0xFAFF, 'CJK Compatibility Ideographs'),
        (0x2F800, 0x2FA1F, 'CJK Compatibility Ideographs Supplement')
    ]
    
    # Check if the character's code point falls within any of the ranges and return the range label
    for start, end, label in unicode_ranges:
        if start <= c <= end:
            return True
    return False

def plot_ci(data, ax, dim=1, **kwargs):
    """
    Plots the mean and confidence interval of the given data on the specified axis.

    Parameters:
    - data: A tensor or array-like object containing the data.
    - ax: The axis object on which to plot the data.
    - dim: The dimension along which to compute the mean and confidence interval.
    - **kwargs: Additional keyword arguments to be passed to the plot function.

    Returns:
    None
    """
    mean = data.mean(dim=dim)
    std = data.std(dim=dim)
    sem95 = 1.96 * std / (len(data)**0.5) 
    ax.plot(range(len(mean)), mean, **kwargs)
    ax.fill_between(range(len(mean)), mean - sem95, mean + sem95, alpha=0.3)
    
plt_params = {'linewidth': 2.2}
def plot_ci_plus_heatmap(data, heat, labels, 
                         color='blue', 
                         linestyle='-',
                         tik_step=10, 
                         method='gaussian', 
                         do_lines=True, 
                         do_colorbar=False, 
                         shift=0.5, 
                         nums = [.99, 0.18, 0.025, 0.6],
                         labelpad=10,
                         plt_params=plt_params):
    
    fig, (ax, ax2) = plt.subplots(nrows=2, sharex=True, gridspec_kw={'height_ratios': [1, 10]}, figsize=(5, 3))
    if do_colorbar:
        fig.subplots_adjust(right=0.8) 
    plot_ci(ax2, data, labels, color=color, linestyle=linestyle, tik_step=tik_step, method=method, do_lines=do_lines, plt_params=plt_params)
    
    y = heat.mean(dim=0)
    x = np.arange(y.shape[0])+1

    extent = [x[0]-(x[1]-x[0])/2. - shift, x[-1]+(x[1]-x[0])/2. + shift, 0, 1]
    img =ax.imshow(y[np.newaxis,:], cmap="plasma", aspect="auto", extent=extent, vmin=0, vmax=14)
    ax.set_yticks([])
    #ax.set_xlim(extent[0], extent[1])
    if do_colorbar:
        cbar_ax = fig.add_axes(nums)  # Adjust these values as needed
        cbar = plt.colorbar(img, cax=cbar_ax)
        cbar.set_label('entropy', rotation=90, labelpad=labelpad)  # Adjust label and properties as needed
    plt.tight_layout()
    return fig, ax, ax2
    
def get_space_char(tokenizer):
    """
    Gets the space character used by the tokenizer.

    Args:
        tokenizer: The tokenizer object used for tokenization.

    Returns:
        str: The space character used by the tokenizer.
    """
    basename = tokenizer.__class__.__name__
    if basename == "QWenTokenizer":
        space_char = " "
    elif basename == "LlamaTokenizer":
        space_char = "▁"
    elif basename == "GPT2Tokenizer":
        space_char = "Ġ"
    else:
        raise NotImplementedError(f"Tokenizer {basename} not implemented")

    return space_char


def raw_tok_to_id(string, tokenizer, add_space=False):
    """
    Converts a string to its corresponding token ID using the provided tokenizer.

    Args:
        string (str): The input string to be converted.
        tokenizer: The tokenizer object used for tokenization.
    Returns:
       The Token ID. If the token ID is None, the string could not be tokenized.

    Raises:
        NotImplementedError: If the tokenizer is not implemented.
    """
    assert type(string) == str, "Input must be a string"
    basename = tokenizer.__class__.__name__
    if tokenizer.is_fast:
        warnings.warn(f"Using {basename} with is_fast = True")


    if add_space:
        space_char = get_space_char(tokenizer)
        string = space_char + string

    if basename == "QWenTokenizer":
        string = string.encode('utf-8')
        try:
            token_id = tokenizer._convert_token_to_id(string)
        except ValueError:
            token_id = None

    elif basename == "LlamaTokenizer" or basename == "GPT2Tokenizer":

        token_id =  tokenizer.convert_tokens_to_ids(string)     
        if token_id == tokenizer.unk_token_id:
            token_id = None

    else:
        raise NotImplementedError(f"Tokenizer {basename} not implemented")

    return token_id, string


def tok_to_id(string, tokenizer):
    """
    Converts a string to its corresponding token ID using the provided tokenizer.
    If adding a space to the front of the string results in a valid token, the space is added.

    Args:
        string (str): The input string to be converted.
        tokenizer: The tokenizer object used for tokenization.
    Returns:
        tuple: A tuple containing the token ID and the modified string.
        If the token ID is None, the string could not be tokenized.
    """
    #     Note:
    #     Zero is not always the unknown token. It depends on the tokenizer implementation.
    #     For Llama-2, Mistral, 0 is <unk>.
    #     For Qwen, 0 is b"!".
    #     For gpt-2 and tinystories, 0 is "!". (unknown is 50256 for gpt-2/tinystories)
        
    space_char = get_space_char(tokenizer)
    
    token_id = raw_tok_to_id(string, tokenizer)
    if token_id is None:
        token_id = raw_tok_to_id(space_char + string, tokenizer)
        return token_id, space_char + string
    else:
        return token_id, string

def colour_encode(string, tokenizer):
    tokens = tokenizer.encode(string, add_special_tokens=False)
    print_tok(tokens, tokenizer)

def get_tok_prefix_ids(string, tokenizer, include_space = False, return_tensor = 'pt'):
    prefixes = [string[:i] for i in range(1, len(string) + 1)]
    if include_space:
        prefixes = [get_space_char(tokenizer) + prefix for prefix in prefixes] + prefixes
    valid_tokens = [x for x in prefixes if x in tokenizer.get_vocab()]
    prefix_ids = tokenizer.convert_tokens_to_ids(valid_tokens)
    if return_tensor == 'pt':
        return torch.tensor(prefix_ids)
    else:
        return prefix_ids
    

# %%
def print_tok(token_ids, tokenizer):
    """
    Print decoded tokens with rotating background colors.

    Parameters:
    - token_ids: List of token IDs to be decoded.
    - tokenizer: An object that has a .decode method for token IDs.
    """
    # List of ANSI escape codes for different background colors
    bg_colors = [
        "\033[41m",  # Red
        "\033[42m",  # Green
        "\033[43m",  # Yellow
        "\033[44m",  # Blue
        "\033[45m",  # Magenta
        "\033[46m",  # Cyan
    ]
    reset = "\033[0m"  # Reset styling

    # Cycle through the background colors as tokens are printed
    color_index = 0

    for token_id in token_ids:
        decoded_token = tokenizer.decode([token_id])
        print(f"{bg_colors[color_index]}{decoded_token}{reset}", end='')
        # Move to the next color, cycling back to the start if necessary
        color_index = (color_index + 1) % len(bg_colors)
