# %%
%load_ext autoreload
%autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, tokenize_suffixes, TokenizedSuffixesResult, find_all_tokens
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.llm import suffix_preamble, run, measure_performance
from utils.plot import plot_ci_simple
from utils.config_argparse import try_parse_args
from src.constants import LANG2NAME, LANG_BANK
from eindex import eindex
from collections import namedtuple
import warnings
import re
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache
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
    src_lang: str = 'fr'
    dest_lang: str = 'zh'
    latent_lang: str = 'en'
    model_name: str = "meta-llama/Llama-2-7b-hf" #'gpt2' #'meta-llama/Meta-Llama-3-8B'
    # single_token_only: bool = False
    # multi_token_only: bool = False
    # out_dir: str = './visuals'
    dataset_path: str = "data/butanium_v2.tsv"
    debug: bool = True
    num_multi_shot : int = 5
    token_add_spaces: bool = True
    token_add_leading_byte: bool = True
    token_add_prefixes : bool = False
    token_add_capitalization : bool = True
    # dataset_filter_correct : bool = True
    # intervention_func : str = 'hook_reject_subspace'
    # log_file : str = 'DUMMY_NAME'
    # metric : str = 'p_alt'
    # metric_goal : str = 'max'
    # use_reverse_lens : bool = False
    # rev_lens_scale : bool = 1
    # only_compute_stats : bool = False
    word_list_key : str = 'claude'
    cache_prefix : bool = True

cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
    
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

# %%
def sanity_check(suffix_words):
    prompt = gen_prompt(src_words = src_words,
                    dest_words = dest_words,
                    src_lang = cfg.src_lang, 
                    dest_lang = cfg.dest_lang,
                    num_examples=cfg.num_multi_shot)
    kv_cache = gen_kv_cache(prompt, model)
    suffixes = gen_common_suffixes(suffix_words,
                                    src_lang = cfg.src_lang,
                                    dest_lang = cfg.dest_lang)
    suffix_toks = tokenize_suffixes(suffixes, model)
    result = run_with_kv_cache(suffix_toks.input_ids, kv_cache, model)
    logits = eindex(result.logits, suffix_toks.indices, "batch [batch] vocab")
    translated = list(zip(df[cfg.src_lang][:10], model.tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1))))
    
    print(tabulate(translated, headers=[f'{cfg.src_lang=}', f'{cfg.dest_lang=}']))
    
sanity_check(suffix_words)
# %%

# %%
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset

# %%

def parse_word_list(s):
    # Remove the outer brackets and split by commas
    try:
        result = literal_eval(s)
        return result
    except:
        warnings.warn(f"Could not parse row: {s}")
        s = s.strip()[1:-1]
        items = re.split(r',\s*', s)
        
        result = []
        for item in items:
            # Remove surrounding quotes if present
            if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                item = item[1:-1]
            # Handle apostrophes within words
            item = item.replace("'", "'")
            result.append(item)
    
        return result

def build_lang_idx(df_col, lang, vocab, **kwargs):
    array = []
    word_list_key = kwargs.get('word_list_key', f'claude')
    for primary, word_list in df[[lang, f'{word_list_key}_{lang}']].values:
        row_list = parse_word_list(word_list)
        row_list.append(primary)
        tokens = [find_all_tokens(x, vocab, **cfg_dict) for x in row_list]
        try:
            idx = torch.unique(torch.cat(tokens))
        except:
            print(f'{i=}')
            print(f'{row=}')
            print(f'{row_list=}')
            print(f'{tokens=}')
            
        array.append(idx)
    return array

LangIdx = namedtuple('LangIdx', ['src_idx', 'dest_idx', 'latent_idx'],
                     defaults=[None, None, None])

def create_lang_idx(df, vocab, **kwargs):
    src_idx = build_lang_idx(df, cfg.src_lang, vocab, **kwargs)
    dest_idx = build_lang_idx(df, cfg.dest_lang, vocab,**kwargs)
    latent_idx = build_lang_idx(df, cfg.latent_lang, vocab,**kwargs)
    
    assert len(src_idx) == len(dest_idx), "Mismatch between src_idx and dest_idx lengths"
    assert len(src_idx) == len(latent_idx), "Mismatch between src_idx and latent_idx lengths"
    
    return LangIdx(src_idx, latent_idx, dest_idx)

# Usage


# %%

class MixedDataset(Dataset):
    def __init__(self, tensors: Tuple[torch.Tensor, ...], tensor_lists: Tuple[List[torch.Tensor], ...]):
        if not tensors and not tensor_lists:
            raise ValueError("Both tuples are empty")
        
        self.tensors = tensors
        self.tensor_lists = tensor_lists
        
        self.length = len(self.tensors[0]) if self.tensors else len(self.tensor_lists[0])
        
        # Ensure all data have the same length
        if not all(len(t) == self.length for t in self.tensors):
            raise ValueError("All tensors must have the same length")
        if not all(len(l) == self.length for l in self.tensor_lists):
            raise ValueError("All tensor lists must have the same length as tensors")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            tuple(tensor[idx] for tensor in self.tensors),
            tuple(tensor_list[idx] for tensor_list in self.tensor_lists)
        )

    @staticmethod
    def collate_fn(batch):
        # Separate fixed-length tensors and variable-length tensor lists
        fixed_tensors, var_tensor_lists = zip(*batch)
        
        # Handle fixed-length tensors
        collated_fixed = tuple(torch.stack(items) for items in zip(*fixed_tensors))
        
        # Handle variable-length tensor lists
        collated_var = tuple(list(items) for items in zip(*var_tensor_lists))
        
        return collated_fixed + collated_var


    
# %%


# %%
@torch.no_grad
def logit_lens_batched(kv_cache : HookedTransformerKeyValueCache, 
                       suffix_toks : TokenizedSuffixesResult,
                       model,
                       lang_idx : LangIdx,
                       tuned_lens = None, 
                       intervention=None,
                       **kwargs):
    
    device = next(model.parameters()).device
    
    batch_size = kwargs.get('batch_size', 1)
    use_tuned_lens = kwargs.get('use_tuned_lens', False)
    use_tuned_lens = kwargs.get('use_tuned_lens', False)
    
    #return_float = kwargs.get('return_float', False)
    
    fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, **kwargs)
    all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    
    dataset = MixedDataset(suffix_toks, lang_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=MixedDataset.collate_fn)

    runner = tqdm(dataloader, total=len(suffix_toks), desc="Computing logits", position=0, leave=True)
    
    all_src_probs = []
    all_latent_probs = []
    all_dest_probs = []
    
    for i, (toks, toks_mask, toks_idx, src_idx, latent_idx, dest_idx) in enumerate(runner):
        _, cache = run_with_kv_cache(toks, kv_cache, model, fwd_hooks = fwd_hooks, names_filter = all_post_resid)
        
        latents = torch.stack(tuple(cache.values()), dim=1) # (batch, num_layers, seq, d_model)
        latents = eindex(latents, toks_idx, "batch num_layers [batch] d_model") # (batch, num_layers, d_model)
        
        if use_tuned_lens:
            approx_logits = model.tuned_lens(latents)
        else:
            approx_logits = model.unembed(latents)
            
        probs = torch.softmax(approx_logits, dim=-1) # (batch, num_layers, vocab)
        
        
        for batch, (src, latent, dest) in enumerate(zip(src_idx, latent_idx, dest_idx)):
            src_probs = probs[batch, :, src].sum(dim=-1)  # (n_layers)
            latent_probs = probs[batch, :, latent].sum(dim=-1)  # (n_layers)
            dest_probs = probs[batch, :, dest].sum(dim=-1)  # (n_layers)
            all_src_probs.append(src_probs)
            all_latent_probs.append(latent_probs)
            all_dest_probs.append(dest_probs)
        
        runner.update(len(toks))
        
    all_src_probs = torch.stack(all_src_probs, dim=-1) #(bs, n)
    all_latent_probs = torch.stack(all_latent_probs, dim=-1) #(bs, n)
    all_dest_probs = torch.stack(all_dest_probs, dim=-1) #(bs, n)
    all_probs = torch.stack([all_src_probs, all_latent_probs, all_dest_probs], dim=0) #(lang, bs, n)
    
    return all_probs

# %%
def run(cfg):
    prompt = gen_prompt(src_words = src_words,
                        dest_words = dest_words,
                        src_lang = cfg.src_lang, 
                        dest_lang = cfg.dest_lang,
                        num_examples=cfg.num_multi_shot)
    kv_cache = gen_kv_cache(prompt, model)
    lang_idx = create_lang_idx(df, model.tokenizer.vocab, **cfg_dict)

    suffixes = gen_common_suffixes(suffix_words,
                                    src_lang = cfg.src_lang,
                                    dest_lang = cfg.dest_lang)
    suffix_toks = tokenize_suffixes(suffixes, model)
    probs =  logit_lens_batched(kv_cache, suffix_toks, model, lang_idx, batch_size = 8)
    return probs

cfg = Config(token_add_prefixes=True, word_list_key="claude")
probs = run(cfg)
# %%
fig, ax = plt.subplots()
plot_ci_simple(probs[0].cpu(), ax, dim=1, label=f'src={cfg.src_lang}')
plot_ci_simple(probs[1].cpu(), ax, dim=1, label=f'latent={cfg.latent_lang}')
plot_ci_simple(probs[2].cpu(), ax, dim=1, label=f'dest={cfg.dest_lang}')
ax.legend()
plt.show()
# %%
#test_prompt(prompt + suffixes[86], "<0xE5>", model)