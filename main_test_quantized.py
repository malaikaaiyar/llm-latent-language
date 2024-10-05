# %%
# %load_ext autoreload
# %autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, find_all_tokens
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.intervention import Intervention
from src.constants import LANG2NAME, LANG_BANK
from src.llm import safe_tokenize
from utils.data import gen_lang_ids, results_dict_to_csv

from utils.plot import plot_ci_simple
from utils.config_argparse import try_parse_args
from utils.data import parse_word_list, gen_lang_ids, gen_ids
from utils.misc import ci

from eindex import eindex
from collections import namedtuple
import warnings
import re
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache
from src.kv_cache import broadcast_kv_cache
from transformer_lens.utils import test_prompt
# Import GPT-2 tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#disable gradients
torch.set_grad_enabled(False)

from ast import literal_eval
from tabulate import tabulate
# %%
@dataclass
class Config:
    seed: int = 42
    model_name: str = "meta-llama/Llama-2-7b-hf" # 'meta-llama/Meta-Llama-3-8B'
    # single_token_only: bool = False
    # multi_token_only: bool = False
    out_dir: str = './out_iclr'
    dataset_path: str = "data/butanium_v2.tsv"
    debug: bool = True
    num_multi_shot : int = 5
    token_add_spaces: bool = True
    token_add_leading_byte: bool = True
    token_add_prefixes : bool = False
    token_add_capitalization : bool = True
    quantize: Optional[str] = None
    word_list_key : str = 'claude'
    src_lang : str = None
    dest_lang : str = None
    latent_lang : str = 'en'

cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
print(cfg_dict)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.makedirs(cfg.out_dir, exist_ok=True)

# %%
#LOAD_MODEL = False
if 'LOAD_MODEL' not in globals():
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device,
                                                            dtype = torch.float16)

    tokenizer = model.tokenizer
    tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore
    LOAD_MODEL = False    
    