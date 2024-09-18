# %%
%load_ext autoreload
%autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.llm import suffix_preamble, run, measure_performance
from utils.config_argparse import try_parse_args
from src.constants import LANG2NAME, LANG_BANK

from ast import literal_eval
# %%
@dataclass
class Config:
    seed: int = 42
    src_lang: str = 'fr'
    dest_lang: str = 'zh'
    latent_lang: str = 'en'
    model_name: str = 'meta-llama/Meta-Llama-3-8B'
    # single_token_only: bool = False
    # multi_token_only: bool = False
    # out_dir: str = './visuals'
    dataset_path: str = "data/butanium_word_translation.csv"
    debug: bool = True
    num_multi_shot : int = 5
    # token_add_spaces: bool = True
    # token_add_leading_byte: bool = False
    # token_add_prefixes : bool = False
    # dataset_filter_correct : bool = True
    # intervention_func : str = 'hook_reject_subspace'
    # log_file : str = 'DUMMY_NAME'
    # metric : str = 'p_alt'
    # metric_goal : str = 'max'
    # use_reverse_lens : bool = False
    # rev_lens_scale : bool = 1
    # only_compute_stats : bool = False
    cache_prefix : bool = True

cfg = Config()
cfg = try_parse_args(cfg)
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
#LOAD_MODEL = False
if 'LOAD_MODEL' not in globals():
    LOAD_MODEL = False
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype = torch.float16)
    tokenizer = model.tokenizer
    tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore    
# %%

df = pd.read_csv(cfg.dataset_path)

# %%

src_words = LANG_BANK[cfg.src_lang]
dest_words = LANG_BANK[cfg.dest_lang]

suffix_words = [literal_eval(x) for x in df[cfg.src_lang]]
# %%
prompt = gen_prompt(src_words = src_words,
                    dest_words = dest_words,
                    src_lang = cfg.src_lang, 
                    dest_lang = cfg.dest_lang,
                    num_examples=cfg.num_multi_shot)

suffixes = gen_common_suffixes(suffix_words,
                                 src_lang = cfg.src_lang,
                                 dest_lang = cfg.dest_lang)
                                    
