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
# %%

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


def process_suffix_toks(suffix_toks, model):
    vocab = model.tokenizer.get_vocab()
    if "Llama-2" in model.cfg.model_name:
        assert torch.all(suffix_toks[:, 0] == vocab["▁"]), "LLama tokenizer should prepend space token"
        suffix_toks = suffix_toks[:, 1:]
        
    elif "gemma" in model.cfg.model_name:
        pass 
    
    else:
        raise ValueError(f"Check {model.cfg.model_name} tokenization first, add case to prefix.py")
    printd(suffix_toks)
    return suffix_toks

def tokenize_suffixes(suffixes : List[str], model):
    device = next(model.parameters()).device
    raw_suffix_toks, attention_mask = model.tokenizer(suffixes, 
                                                    add_special_tokens=False, 
                                                    return_tensors="pt", 
                                                    padding = True).values() #remove start of sequence character
    good_idx = torch.ones(len(raw_suffix_toks), dtype=torch.bool)
    if torch.any(attention_mask == 0):
        print("Warning: tokenize_suffixes - Attention mask has zeros")
        # assume that most common number of tokens is correct
        # we only get more tokens if something screwed up and a chinese character was sampled
        # when it shouldn't have, so take the ids with shortest attention mask
        counts = attention_mask.sum(dim=1)
        correct_count = torch.mode(counts, dim=0).values
        good_idx = counts == correct_count
        # global bad_idx
        # bad_idx = torch.where(counts != correct_count)
        
        suffix_toks = raw_suffix_toks[good_idx][:, :correct_count]
        assert torch.all(attention_mask[good_idx][:, :correct_count] == 1), "Some tokens have zero attention mask"
    else:
        suffix_toks = raw_suffix_toks
    
    suffix_toks = suffix_toks.to(device)
    suffix_toks = process_suffix_toks(suffix_toks, model)
    return suffix_toks, good_idx
    
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
    
    with model.hooks(fwd_hooks, names_filter):
        if names_filter == []:
            logits = model(tokens, past_kv_cache=kv_cache)
            return RunWithKVCacheResult(logits=logits, cache=None)
        else:
            logits, cache = model.run_with_cache(tokens, past_kv_cache=kv_cache, names_filter=names_filter)
            return RunWithKVCacheResult(logits=logits, cache=cache)    
    
def batched_predict_next(kv_cache : HookedTransformerKeyValueCache,
                           suffixes_toks : Int[Tensor, "batch seq"],
                           model,
                           fwd_hooks = [],
                           hooks_filter = [], 
                           batch_size = 1,
                           **kwargs):
    
    desc = kwargs.get("desc", "")
    position = kwargs.get("position", 0)
    leave = kwargs.get("leave", True)
    # assume all suffixes tokenize to the same number of tokens
    all_probs = []
    all_toks = []

    suffix_toks_batched = torch.split(suffixes_toks, batch_size, dim=0)
    
    runner = tqdm(suffix_toks_batched, total=len(suffixes_toks), desc=desc, position=position, leave=leave)
    
    for batch in runner:
        logits = run_with_kv_cache(batch, kv_cache, model, fwd_hooks, hooks_filter).logits[:, -1].detach()
        probs = torch.softmax(logits, dim=-1)
        max_probs, max_tokens = torch.max(probs, dim=-1)
        
        all_probs.append(max_probs)
        all_toks.append(max_tokens)
        runner.update(len(batch))
        
    all_probs = torch.cat(all_probs, dim=0)
    all_toks = torch.cat(all_toks, dim=0)
    return all_probs, all_toks

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
    
    prompt = gen_data.generate_translation_prompt(None, src_lang, dest_lang)
    kv_cache = prefix.gen_kv_cache(prompt, model)
    suffixes = gen_data.generate_common_suffixes(df[src_lang], **kwargs)
    suffix_toks, _ = prefix.tokenize_suffixes(suffixes, model)
    
    target_toks = torch.LongTensor(df[f'{dest_lang}_tok']).to(device)
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
