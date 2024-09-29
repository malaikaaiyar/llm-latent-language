# %%
%load_ext autoreload
%autoreload 2
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
    word_list_key : str = 'claude'
    src_lang : str = None
    dest_lang : str = None
    latent_lang : str = 'en'

cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
    
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
# %%
df = pd.read_csv(cfg.dataset_path, delimiter = '\t') 



# %%

    
def good_batched_multi_token_only_end(prompt = None,
                                      model = None,
                                      suffixes = None, 
                                    intervention = None,
                                    id_bank = None,
                                    cfg=None,
                                    fast = True,
                                    use_alt_latent = False):
    """ 
    known good latent lang plot batched with measuring other tokens
    """
    from src.kv_cache import broadcast_kv_cache
    # prompt_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1) # flush cache
    # model(prompt, past_kv_cache = prompt_cache) #fill kv_cache
    # kv_cache.freeze()
    
    latent_idx = id_bank[cfg.latent_lang]
    dest_idx = id_bank[cfg.dest_lang]
    src_idx = id_bank[cfg.src_lang]
    
    if use_alt_latent:
        alt_latent_idx = torch.empty_like(latent_idx)
        alt_latent_idx[:-1] = latent_idx[1:]
        alt_latent_idx[-1] = latent_idx[0] #alt_latent_idx is rotated by 1
        interv_idx = alt_latent_idx
    else:
        interv_idx = latent_idx

    prompt_cache = gen_kv_cache(prompt, model)
    broadcast_kv_cache(prompt_cache, len(suffixes))

    suffix_toks = safe_tokenize(suffixes, model)

    if fast:
        rejection_subspace = model.unembed.W_U.T[interv_idx]
        rejection_subspace[interv_idx==0] = 0
    else:
        rejection_subspace = None

    #intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers))
    fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, 
                                       rejection_subspaces = rejection_subspace,
                                       latent_idx = interv_idx, 
                                       suffix_idx = suffix_toks.indices)
    
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(suffix_toks.input_ids, past_kv_cache = prompt_cache) # (batch, seq vocab)
        
    logits_last = eindex(logits, suffix_toks.indices, "batch [batch] dmodel")
    probs_last = torch.softmax(logits_last, dim=-1)
    
    probs_dest = eindex(probs_last, dest_idx, "batch [batch num_correct]").cpu().sum(-1) 
    probs_latent = eindex(probs_last, latent_idx, "batch [batch num_correct]").cpu().sum(-1)
    return ci(probs_dest)

# %%
from itertools import permutations


id_bank = gen_lang_ids(df, model, ['en', 'zh', 'fr', 'es', 'de', 'ru'])
combos_pairs = list(permutations(id_bank.keys(), 2))    

results = {}
runner = tqdm(combos_pairs)
for (src_lang, dest_lang) in runner:
    
    cfg_ex = Config(src_lang = src_lang, dest_lang = dest_lang)
    prompt = gen_prompt(src_lang=src_lang, dest_lang=dest_lang)
    src_words = df[cfg_ex.src_lang]
    suffixes = gen_common_suffixes(src_words,
                            src_lang = cfg_ex.src_lang,
                            dest_lang = cfg_ex.dest_lang)
    
    avg_prob, sem95_prob = good_batched_multi_token_only_end(prompt = prompt, 
                                        model = model, 
                                        suffixes = suffixes,
                                        intervention = None,
                                        id_bank = id_bank,
                                        cfg=cfg_ex)
    results[(src_lang, dest_lang)] = (avg_prob.item(), sem95_prob.item())
    runner.set_description(f"Running {src_lang} -> {dest_lang}: {avg_prob.item():.2f} ± {sem95_prob.item():.2f}")
# %%
combos_latent = list(permutations(id_bank.keys(), 3))
print("Running experiments latent and alt...")

results_interv = {}
intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers))

for use_alt_latent in [True, False]:
    runner = tqdm(combos_latent)
    for (latent_lang, src_lang, dest_lang) in runner:
        cfg_ex = Config(src_lang = src_lang, dest_lang = dest_lang, latent_lang = latent_lang)
        prompt = gen_prompt(src_lang=src_lang, dest_lang=dest_lang)
        src_words = df[cfg_ex.src_lang]
        suffixes = gen_common_suffixes(src_words,
                                src_lang = cfg_ex.src_lang,
                                dest_lang = cfg_ex.dest_lang)
        
        
        avg_prob, sem95_prob = good_batched_multi_token_only_end(prompt = prompt, 
                                            model = model, 
                                            suffixes = suffixes,
                                            intervention = intervention,
                                            id_bank = id_bank,
                                            use_alt_latent=use_alt_latent,
                                            cfg=cfg_ex)
        
        results_interv[(use_alt_latent, src_lang, dest_lang, latent_lang)] = (avg_prob.item(), sem95_prob.item())
        if use_alt_latent:
            runner.set_description(f"Running {src_lang} -> {dest_lang} with ALT {latent_lang}: {avg_prob.item():.2f} ± {sem95_prob.item():.2f}")
        else:
            runner.set_description(f"Running {src_lang} -> {dest_lang} {latent_lang}: {avg_prob.item():.2f} ± {sem95_prob.item():.2f}")
# %%
results_interv_no_alt = {k[1:]: v for k, v in results_interv.items() if not k[0]}
results_interv_alt = {k[1:]: v for k, v in results_interv.items() if k[0]}
# %%
results_dict_to_csv