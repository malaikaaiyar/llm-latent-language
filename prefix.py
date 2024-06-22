# %%
import torch
from dq_utils import printd
from typing import List, Tuple, Callable
from jaxtyping import Int
from tqdm.auto import tqdm
from transformer_lens import HookedTransformerKeyValueCache, HookedTransformer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import gen_data
import prefix
from collections import namedtuple
#from gen_data import lang2name
# %%

lang2name = {'fr': 'Français', 'de': 'Deutsch', 'en': 'English', 'zh': '中文'}

__DEBUG__ = True

def broadcast_kv_cache(kv_cache : HookedTransformerKeyValueCache, n : int):
    """
    Broadcasts the key-value kv_cache for parallel processing, reshaping its elements
    from (a, b, c, d) to (n, b, c, d), assuming all elements in dimension 'a' are identical
    and can be replicated to dimension 'n'.

    Args:
        kv_cache (object): The key-value cache object.
        n (int): The number of parallel processes.

    Returns:
        None
    """
    for e in kv_cache:
        if e.past_keys.dim() == 4 and e.past_keys.size(0) > 1:
            # Assuming the first dimension has redundant copies, we take one and expand it
            e.past_keys = e.past_keys[0].unsqueeze(0).expand(n, -1, -1, -1)
            e.past_values = e.past_values[0].unsqueeze(0).expand(n, -1, -1, -1)
        else:
            # If already in correct form or not expanded, simply adjust the dimensions
            e.past_keys = e.past_keys.expand(n, -1, -1, -1)
            e.past_values = e.past_values.expand(n, -1, -1, -1)
    if kv_cache.previous_attention_mask.dim() == 2 and kv_cache.previous_attention_mask.size(0) > 1:
        # Similarly adjust the attention mask
        kv_cache.previous_attention_mask = kv_cache.previous_attention_mask[0].unsqueeze(0).expand(n, -1)
    else:
        kv_cache.previous_attention_mask = kv_cache.previous_attention_mask.expand(n, -1)

    
def gen_kv_cache(prompt : str | Int[Tensor, "batch seq"] | Int[Tensor, "seq"],
                 model : HookedTransformer
) -> HookedTransformerKeyValueCache:
    device = next(model.parameters()).device
    if isinstance(prompt, str):
        prompt = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
    else:
        prompt = prompt.to(device)
    if prompt.dim() == 1:
        bs = 1
    else:
        bs = prompt.size(0)
        
    kv_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, bs) # flush cache
    model(prompt, past_kv_cache = kv_cache) #fill kv_cache
    kv_cache.freeze()
    return kv_cache
    
RunWithKVCacheResult = namedtuple('RunWithKVCacheResult', ['logits', 'cache'], defaults=[None])
    
def run_with_kv_cache(tokens : Int[Tensor, "batch seq"],
                    kv_cache : HookedTransformerKeyValueCache,
                    model : HookedTransformer,
                    fwd_hooks : List[Callable] = [],
                    names_filter : List[str] = [],
) -> Tuple[Tensor, Tensor]:
    device = next(model.parameters()).device
    kv_cache.freeze()
    broadcast_kv_cache(kv_cache, len(tokens))
    tokens = tokens.to(device)
    
    if names_filter == []:
        logits = model(tokens, past_kv_cache=kv_cache)
        return RunWithKVCacheResult(logits=logits, cache=None)
    else:
        with model.hooks(fwd_hooks = fwd_hooks):
            logits, cache = model.run_with_cache(tokens, past_kv_cache=kv_cache, names_filter=names_filter)
            return RunWithKVCacheResult(logits=logits, cache=cache)    
    
def batched_predict_next(kv_cache : HookedTransformerKeyValueCache,
                           suffixes_toks : Int[Tensor, "batch seq"],
                           model,
                           fwd_hooks = [],
                           hooks_filter = [], 
                           batch_size = 1,
                           return_logits = False,
                           **kwargs):
    
    desc = kwargs.get("desc", "")
    position = kwargs.get("position", 0)
    leave = kwargs.get("leave", True)
    # assume all suffixes tokenize to the same number of tokens
    all_outs = []
    all_toks = []

    suffix_toks_batched = torch.split(suffixes_toks, batch_size, dim=0)
    
    runner = tqdm(suffix_toks_batched, total=len(suffixes_toks), desc=desc, position=position, leave=leave)
    
    for batch in runner:
        logits = run_with_kv_cache(batch, kv_cache, model, fwd_hooks, hooks_filter).logits[:, -1].detach()
        if return_logits:
            outs = logits
        else:
            outs = torch.softmax(logits, dim=-1)
        max_outs, max_tokens = torch.max(outs, dim=-1)
        
        all_outs.append(max_outs)
        all_toks.append(max_tokens)
        runner.update(len(batch))
        
    all_outs = torch.cat(all_outs, dim=0)
    all_toks = torch.cat(all_toks, dim=0)
    return all_outs, all_toks

SuffixPreambleReturn = namedtuple('SuffixPreambleReturn', ['kv_cache', 'tokens', "indices"])
from IPython.core.debugger import set_trace

def process_suffix_toks(suffix_toks, model):
    vocab = model.tokenizer.get_vocab()
    try:
        if "Llama-2" in model.cfg.model_name:
            #assert torch.all( ((suffix_toks[:, 0] == vocab["▁"]) | (suffix_toks[:, 0] == vocab["▁<"]))), "LLama tokenizer should prepend space token"
            suffix_toks = suffix_toks[:, 1:] # remove space token
            
        elif "gemma" in model.cfg.model_name: # gemma tokenizer does not prepend space token automatically
            pass  
        
        else:
            raise ValueError(f"Check {model.cfg.model_name} tokenization first, add case to prefix.py")
        printd(suffix_toks)
        return suffix_toks
    except AssertionError as e:
        print(e)
        set_trace()  # Start the debugger
        
TokenizeSuffixesResult = namedtuple('TokenizeSuffixes', ['tokens', 'indices'])

def tokenize_suffixes(suffixes : List[str], model):
    device = next(model.parameters()).device
    raw_suffix_toks, attention_mask = model.tokenizer(suffixes, 
                                                    add_special_tokens=False, 
                                                    return_tensors="pt", 
                                                    padding = True).values() #remove start of sequence character
    attention_mask = attention_mask.to(device)
    good_idx = torch.ones(len(raw_suffix_toks), dtype=torch.bool, device=device)
    if torch.any(attention_mask == 0):
        print("Warning: tokenize_suffixes - Attention mask has zeros")
        # assume that most common number of tokens is correct
        # we only get more tokens if something screwed up and a chinese character was sampled
        # when it shouldn't have, so take the ids with shortest attention mask
        counts = attention_mask.sum(dim=1)
        correct_count = torch.mode(counts, dim=0).values
        good_idx = (counts == correct_count).to(device)
        # global bad_idx
        # bad_idx = torch.where(counts != correct_count)
        
        suffix_toks = raw_suffix_toks[:, :correct_count]
        assert torch.all(attention_mask[good_idx][:, :correct_count] == 1), "Incorrect attention mask trimming"
        # assert not torch.all(attention_mask[good_idx][:, :correct_count+1] == 1), "Incorrect attention mask trimming"
    else:
        suffix_toks = raw_suffix_toks
    
    suffix_toks = suffix_toks.to(device)
    suffix_toks = process_suffix_toks(suffix_toks, model)
    assert suffix_toks.shape[0] == len(suffixes), "Suffixes and tokens should have the same length"
    return TokenizeSuffixesResult(suffix_toks, good_idx)

def suffix_preamble(src_words, model, keep_idx, src_lang = None, dest_lang = None, **kwargs):
    assert src_lang in lang2name and dest_lang in lang2name, "src_lang and dest_lang must be provided"
    prompt = gen_data.generate_translation_prompt(None, src_lang, dest_lang)
    kv_cache = prefix.gen_kv_cache(prompt, model)
    suffixes = gen_data.generate_common_suffixes(src_words, src_lang, dest_lang)
    suffix_toks, keep_idx = prefix.tokenize_suffixes(suffixes, model)
    
    return SuffixPreambleReturn(kv_cache, suffix_toks, keep_idx)


def run(src_words, model, src_lang, dest_lang, batch_size = None, **kwargs):    
    assert batch_size is not None, "prefix.run: batch_size must be provided"
    kv_cache, suffix_toks, keep_idx = prefix.suffix_preamble(src_words, model, src_lang, dest_lang)
    all_probs, all_toks = prefix.batched_predict_next(kv_cache, suffix_toks, model, 
                                                      batch_size=batch_size, desc=f"{src_lang} -> {dest_lang}")
    
    unk_id = model.tokenizer.convert_tokens_to_ids('<unk>')
    mask_all_probs = -torch.ones_like(all_probs)
    mask_all_toks = torch.zeros_like(all_toks) + unk_id
    mask_suffix_toks = torch.zeros_like(suffix_toks[:, 0]) + unk_id
    
    mask_all_probs[keep_idx] = all_probs[keep_idx]
    mask_all_toks[keep_idx] = all_toks[keep_idx]
    mask_suffix_toks[keep_idx] = suffix_toks[keep_idx][:, 0]
    
    # Keep outputs the same size, but mask out the ones that are not in the vocab
    # keep_idx = boolean_mask of which ones are in the vocab
    return mask_all_probs, mask_all_toks, mask_suffix_toks, keep_idx


@torch.no_grad()
def measure_performance(df, model, **kwargs): 
    src_lang = kwargs.get("src_lang", None)
    dest_lang = kwargs.get("dest_lang", None)
    batch_size = kwargs.get("batch_size", 1)
    device = next(model.parameters()).device
    assert src_lang is not None and dest_lang is not None, "src_lang and dest_lang must be provided"
    correct = 0
    processed = 0
    total_loss = 0
    tokenizer = model.tokenizer
    
    kv_cache, suffix_toks, _ = suffix_preamble(df[src_lang], model, src_lang, dest_lang)
    
    target_toks = torch.LongTensor(list(df[f'{dest_lang}_tok'])).to(device)
    tensor_dataset = TensorDataset(suffix_toks, target_toks)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size)
    runner = tqdm(dataloader, total=len(dataloader), desc="Measuring performance", position=0, leave=True)
    
    loss = torch.nn.CrossEntropyLoss()
    
    for i, (suffix, target) in enumerate(runner):
        logits = prefix.run_with_kv_cache(suffix, kv_cache, model).logits[:, -1].detach()
        total_loss += loss(logits, target)
        correct += (logits.argmax(dim=-1) == target).sum()
        processed += len(target)
        runner.set_description(f"Accuracy: {correct.item() / processed:.3f}, Loss: {total_loss.item() / (i+1):.3f}")
    return correct / len(df), total_loss / len(dataloader)

# %%
