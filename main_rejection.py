# %%
%load_ext autoreload
%autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, tokenize_suffixes, TokenizedSuffixesResult
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.llm import suffix_preamble, run, measure_performance
from utils.config_argparse import try_parse_args
from src.constants import LANG2NAME, LANG_BANK
from eindex import eindex
from collections import namedtuple

from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache
# Import GPT-2 tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#disable gradients
torch.set_grad_enabled(False)

from ast import literal_eval
# %%
@dataclass
class Config:
    seed: int = 42
    src_lang: str = 'fr'
    dest_lang: str = 'de'
    latent_lang: str = 'en'
    model_name: str = "meta-llama/Llama-2-7b-hf" #'gpt2' #'meta-llama/Meta-Llama-3-8B'
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

# handelled by tokenize_suffixes
# if "Llama-2" in cfg.model_name:
#     test_suffixes2 = ["🌍" + x for x in suffixes]
#     raw_suffix_toks = tokenize_suffixes(test_suffixes2, model)
#     space_token_id = model.tokenizer.convert_tokens_to_ids("▁")
#     earth_id = model.tokenizer.convert_tokens_to_ids("🌍")
#     #print(raw_suffix_toks)     
#     assert torch.all(raw_suffix_toks.input_ids[:, 0] == space_token_id), "llama2 has leading space token"
#     assert torch.all(raw_suffix_toks.input_ids[:, 1] == earth_id), "llama2 single token for 🌍"
#         # they add leading spaces :'(
#     new_suffix_toks = raw_suffix_toks.input_ids[:, 2:]
#     new_attention_mask = raw_suffix_toks.attention_mask[:, 2:]
#     new_idx = raw_suffix_toks.indices - 2
#     # suffix_toks.attention_mask = suffix_toks.attention_mask[:,1:]
#     suffix_toks = TokenizedSuffixesResult(input_ids=new_suffix_toks, 
#                                             attention_mask=new_attention_mask, 
#                                             indices=new_idx)

suffix_toks = tokenize_suffixes(suffixes, model)
# %%
                             
result = run_with_kv_cache(suffix_toks.input_ids, kv_cache, model)
logits = eindex(result.logits, suffix_toks.indices, "batch [batch] vocab")
translated = list(zip(df[cfg.src_lang][:10], model.tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1))))
from tabulate import tabulate
print(tabulate(translated, headers=[f'{cfg.src_lang=}', f'{cfg.dest_lang=}']))
# %%

# %%
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset


def token_idx(word : str, tokenizer, is_llama2 = False) -> torch.Tensor:
    # strip white space
    word = word.strip().lower()
    wordlist_no_space = [word, word.lower(), word.capitalize(), word.upper()]
    wordlist_with_space = [f" {word}" for word in wordlist_no_space]
    wordlist = wordlist_no_space + wordlist_with_space
    print(wordlist)
    if is_llama2:
        wordlist = "🌍" + word
        model.tokenizer
# %%



@dataclass
class LangIdx:
    src_idx: List[torch.Tensor]
    dest_idx: List[torch.Tensor]
    latent_idx: List[torch.Tensor]

class VariableLengthDataset(Dataset):
    def __init__(self, suffix_toks: torch.Tensor, lang_idx: LangIdx):
        self.suffix_toks = suffix_toks
        self.lang_idx = lang_idx
        
        assert len(suffix_toks) == len(lang_idx.src_idx), "Mismatch between suffix_toks and src_idx lengths"
        assert len(suffix_toks) == len(lang_idx.dest_idx), "Mismatch between suffix_toks and dest_idx lengths"
        assert len(suffix_toks) == len(lang_idx.latent_idx), "Mismatch between suffix_toks and latent_idx lengths"

    def __len__(self):
        return len(self.suffix_toks)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.suffix_toks[idx],
            self.lang_idx.src_idx[idx],
            self.lang_idx.dest_idx[idx],
            self.lang_idx.latent_idx[idx]
        )

    @staticmethod
    def collate_fn(batch):
        suffix_toks, src_idx, dest_idx, latent_idx = zip(*batch)
        suffix_toks = torch.stack(suffix_toks)
        return suffix_toks, list(src_idx), list(dest_idx), list(latent_idx)


@torch.no_grad
def logit_lens_batched(kv_cache : HookedTransformerKeyValueCache, 
                       suffix_toks : Int[Tensor, "batch seq2"],
                       model,
                       lang_idx : LangIdx,
                       tuned_lens = None, 
                       intervention=None,
                       **kwargs):
    
    device = next(model.parameters()).device
    
    batch_size = kwargs.get('batch_size', 1)
    use_tuned_lens = kwargs.get('use_tuned_lens', False)
    use_tuned_lens = kwargs.get('use_tuned_lens', False)
    
    device = next(model.parameters()).device
    lang_idx = lang_idx.to(device)
    suffix_toks = suffix_toks.to(device)
    
    #return_float = kwargs.get('return_float', False)
    
    fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, **kwargs)
    all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    
    dataset = VariableLengthDataset(suffix_toks, lang_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    runner = tqdm(dataloader, total=len(suffix_toks), desc="Computing logits", position=0, leave=True)
    
    all_src_probs = []
    all_dest_probs = []
    all_latent_probs = []
    
    for i, (toks, src_idx, latent_idx, dest_idx) in enumerate(runner):
        cache = run_with_kv_cache(toks, kv_cache, model, fwd_hooks = fwd_hooks, names_filter = all_post_resid)
        
        latents = [act[:, -1, :] for act in cache.values()] # List of (batch, d_model)
        latents = torch.stack(latents, dim=1) # (batch, num_layers, d_model)
        
        if use_tuned_lens:
            approx_logits = model.tuned_lens(latents)
        else:
            approx_logits = model.unembed(latents)
            
        probs = torch.softmax(approx_logits, dim=-1)
        
        src_probs = [probs[i, :, src_idx[i]].sum(dim=-1) for i in range(len(src_idx))]
        latent_probs = [probs[i, :, latent_idx[i]].sum(dim=-1) for i in range(len(latent_idx))]
        dest_probs = [probs[i, :, dest_idx[i]].sum(dim=-1) for i in range(len(dest_idx))]
        #probs = eindex(probs, idx, "bs n_layer [bs lang] -> lang n_layer bs")
        all_src_probs.append(src_probs)
        all_latent_probs.append(latent_probs)
        all_dest_probs.append(dest_probs)
        runner.update(len(toks))
    all_src_probs = torch.cat(all_src_probs, dim=-1)
    all_latent_probs = torch.cat(all_latent_probs, dim=-1)
    all_dest_probs = torch.cat(all_dest_probs, dim=-1)
    all_probs = torch.stack([all_src_probs, all_latent_probs, all_dest_probs], dim=0) #(lang, bs, n)
    
    return all_probs
# %%
