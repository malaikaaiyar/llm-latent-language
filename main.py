# %%
try:
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
except:
    pass
# %%
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm.auto import tqdm
import pickle 
import argparse
import logging

logging.basicConfig(level=logging.INFO)
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
from tuned_lens_wrap import load_tuned_lens
#from reverse_tuned_lens import ReverseLens
import dq_utils
from logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap
import intervention
from intervention import Intervention
from config_argparse import try_parse_args
# %%

@dataclass
class Config:
    seed: int = field(
        default=42,
        metadata={"help": "Random seed for reproducibility."}
)
    src_lang: str = field(
        default='fr',
        metadata={"help": "Source language for translation."}
)
    dest_lang: str = field(
        default='zh',
        metadata={"help": "Destination language for translation."}
)
    latent_lang: str = field(
        default='en',
        metadata={"help": "Intermediate (laten) language used in the model."}
)
    model_name: str = field(
        default='gemma-2b',
        metadata={"help": "Name or path of the model [gemma-2b | Llama-2-7b-hf]."}
)
    single_token_only: bool = field(
        default=False,
        metadata={"help": "Process only single tokens if True."}
)
    multi_token_only: bool = field(
        default=False,
        metadata={"help": "Process only multi tokens if True."}
)
    dataset_path: str = field(
        default="./data/synth_gemma_2b",
        metadata={"help": "Path to the dataset used."}
)
    debug: bool = field(
        default=True,
        metadata={"help": "Enable debug messages."}
)
    num_multi_shot: int = field(
        default=5,
        metadata={"help": "Number of shots for translation prompt."}
)
    token_add_spaces: bool = field(
        default=True,
        metadata={"help": "Add leading spaces to tokens if True."}
)
    token_add_leading_byte: bool = field(
        default=False,
        metadata={"help": "Add a leading byte to tokens if True."}
)
    token_add_prefixes: bool = field(
        default=False,
        metadata={"help": "Add prefixes to tokens if True."}
)
    dataset_filter_correct: bool = field(
        default=True,
        metadata={"help": "Filter dataset to tokens that translate between src_lang <-> dest_lang."}
)
    use_tuned_lens: bool = field(
        default=False,
        metadata={"help": "Use tuned lens instead of logit lens."}
)
    interv_match_latent: bool = field(
        default=True,
        metadata={"help": "Apply interventions to the correct word in the latent space if True.\
            Else, use unrelated word."}
)
    interv_steer_coeff: float = field(
        default=1.0,
        metadata={"help": "Scaling factor for steering intervention."}
)
    start_layer_low: int = field(
        default=0,
        metadata={"help": "Lowest layer to start interventions sweep."}
)
    start_layer_high: int = field(
        default=-1,
        metadata={"help": "Highest layer to start intervention sweep."}
)
    end_layer_low: int = field(
        default=0,
        metadata={"help": "Lowest layer to end intervention sweep."}
)
    end_layer_high: int = field(
        default=-1,
        metadata={"help": "Highest layer to end intervention sweep."}
)
    intervention_func: str = field(
        default='hook_reject_subspace',
        metadata={"help": "Function to use for interventions (see intervention.p)."}
)
    log_file: str = field(
        default='DUMMY_NAME',
        metadata={"help": "Basename for log files."}
)
    batch_size: int = field(
        default=32,
        metadata={"help": "Batch size for processing."}
)
#     metric: str = field(
#         default='p_alt',
#         metadata={"help": "Metric to optimize for [p_alt | p_out | lp_diff]."}
# )
#     metric_goal: str = field(
#         default='max',
#         metadata={"help": "Goal for the optimization (max or mi)."}
# )
    # rev_lens_scale: float = field(
    #     default=2,
    #     metadata={"help": "Scale factor for reverse lens."}
    #)
    only_compute_stats: bool = field(
        default=True,
        metadata={"help": "Compute only statistics if True."}
)
    trans_thresh: float = field(
        default=0.5,
        metadata={"help": "Threshold for translation quality."}
)
LOAD_MODEL = False # only for debugging
cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
# %%
# fix random seed
seed = cfg.seed #42
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_grad_enabled(False)
# %%
# pd.set_option('display.max_rows', 100)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns
# pd.set_option('display.width', None)  # Auto-detect the display width for wrapping
# pd.set_option('display.max_colwidth', None)  # Show full length of data in columns

# %%
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
# tokenizer_vocab = tokenizer.get_vocab()
# %%
if 'LOAD_MODEL' not in globals():
    
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype = torch.float16)
    tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore
    if cfg.use_tuned_lens:
        tuned_lens = load_tuned_lens(model).float()
        model.tuned_lens = tuned_lens
    # if cfg.use_reverse_lens:
    #     reverse_lens = ReverseLens.from_tuned_lens(tuned_lens)
    #     model.reverse_lens = reverse_lens
    LOAD_MODEL = False

# %%
def main(dataset, cfg):

    layer_log2 = {}
    info = {}
    start_lower, start_upper = cfg.start_layer_low, cfg.start_layer_high
    end_lower, end_upper = cfg.end_layer_low, cfg.end_layer_high
    #interv_steer_coeff_list = [0.9, 1.0, 1.01, 1.02, 1.05, 1.3, 1.5][::-1]
    #interv_steer_coeff_list = [1.0]

    total_iterations = dq_utils.calculate_iterations(start_lower, start_upper, end_lower, end_upper)
    outer_pbar = tqdm(total=total_iterations, desc='Overall Progress', leave=True)

    import intervention
    from logit_lens import get_logits, plot_logit_lens_latents

    # if cfg.metric_goal == 'max':
    #     best_stats = {cfg.metric: -np.inf}
    # else:
    #     best_stats = {cfg.metric: np.inf}
        
    for start_layer in range(start_lower,start_upper):
        for end_layer in range(end_lower, end_upper):
            if start_layer >= end_layer:
                continue
            
            interv = Intervention(cfg.intervention_func, range(start_layer, end_layer))
            latent_diff, logits_diff = get_logits(dataset, model, intervention=interv,  **cfg_dict)
            latent_diff = latent_diff.float()
            logits_diff = logits_diff.float()
            stats = plot_logit_lens_latents(logits_diff, dataset, **cfg_dict, title="diff", cfg=cfg)
            
            outer_pbar.set_description(f"Trying: {dq_utils.str_dict(stats)}")
            outer_pbar.update(1)  # Increment the progress bar after each inner iteration
            layer_log2[(start_layer, end_layer)] = stats

    info ={
        "interv_desc" : interv.description,
        "len correct dataset" : len(dataset),
    }

    outer_pbar.close()  # Ensure to close the progress bar after the loop completes
    # Save layer_log2 to a pickle file
    return layer_log2, info

# %%
# df_src = pd.read_csv(os.path.join(cfg.dataset_path, cfg.src_lang, 'clean.csv')).reindex()
# df_dest = pd.read_csv(os.path.join(cfg.dataset_path, cfg.dest_lang, 'clean.csv')).reindex()
# df_raw_data = gen_data.merge_datasets(df_src, df_dest, tokenizer_vocab, cfg)

# %%
%load_ext autoreload
%autoreload 2
# %%
import gen_data
prompt = gen_data.generate_translation_prompt(None, cfg.src_lang, cfg.dest_lang)
raw_dataset = gen_data.load_dataset(cfg.dataset_path, cfg.src_lang, cfg.dest_lang, cfg.latent_lang)
raw_dataset = gen_data.remove_dups(raw_dataset)
dataset = gen_data.keep_correct(raw_dataset, prompt, cfg) # TODO
#hf_model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=hf_token, load_in_8bit=True)
measure_performance(correct_dataset, model) #TODO fix

# %%
if False:
    layer_log2, info = main(dataset, cfg)
    dq_utils.write_log(layer_log2, cfg, info)
    