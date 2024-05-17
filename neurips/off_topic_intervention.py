import sys
import os

# Get the parent directory path
parent_dir = os.path.dirname(os.path.abspath(__file__))

# Add the parent directory to the Python path
sys.path.append(parent_dir)

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

