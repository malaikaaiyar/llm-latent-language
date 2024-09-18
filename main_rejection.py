# %%
%load_ext autoreload
%autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, tokenize_suffixes
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.llm import suffix_preamble, run, measure_performance
from utils.config_argparse import try_parse_args
from src.constants import LANG2NAME, LANG_BANK
from eindex import eindex

from transformers import AutoTokenizer
# Import GPT-2 tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")

from ast import literal_eval
# %%
@dataclass
class Config:
    seed: int = 42
    src_lang: str = 'fr'
    dest_lang: str = 'de'
    latent_lang: str = 'en'
    model_name: str = 'gpt2' #'meta-llama/Meta-Llama-3-8B'
    # single_token_only: bool = False
    # multi_token_only: bool = False
    # out_dir: str = './visuals'
    dataset_path: str = "data/butanium_v2.tsv"
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
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype = torch.float16)
    tokenizer = model.tokenizer
    tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore
    LOAD_MODEL = False    
# %%
df = pd.read_csv(cfg.dataset_path, delimiter = '\t') 

# %%

src_words = LANG_BANK[cfg.src_lang]
dest_words = LANG_BANK[cfg.dest_lang]
suffix_words = df[cfg.src_lang]
# %%
prompt = gen_prompt(src_words = src_words,
                    dest_words = dest_words,
                    src_lang = cfg.src_lang, 
                    dest_lang = cfg.dest_lang,
                    num_examples=cfg.num_multi_shot)
kv_cache = gen_kv_cache(prompt, model)


suffixes = gen_common_suffixes(suffix_words,
                                 src_lang = cfg.src_lang,
                                 dest_lang = cfg.dest_lang)
# %%
test_suffixes = suffixes[:10]
suffix_toks = tokenize_suffixes(test_suffixes, model)                                    
result = run_with_kv_cache(suffix_toks.input_ids, kv_cache, model)
logits = eindex(result.logits, suffix_toks.indices, "batch [batch] vocab")
translated = list(zip(df[cfg.src_lang][:10], model.tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1))))
print(translated)
# %%
from tabulate import tabulate
print(tabulate(translated, headers=[f'{cfg.src_lang=}', f'{cfg.dest_lang=}']))
# %%
