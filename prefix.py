# %%
import torch
from dq_utils import printd
from typing import List
from tqdm.auto import tqdm
from transformer_lens import HookedTransformerKeyValueCache, HookedTransformer

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


def process_suffix_toks(suffix_toks):
    if "Llama-2" in model.cfg.model_name:
        assert torch.all(suffix_toks[:, 0] == vocab["▁"]), "LLama tokenizer should prepend space token"
        suffix_toks = suffix_toks[:, 1:]
        
    elif "gemma" in model.cfg.model_name:
        pass 
    
    else:
        raise ValueError(f"Check {model.cfg.model_name} tokenization first, add case to prefix.py")
    printd(suffix_toks)
    return suffix_toks

def tokenize_suffixes(suffixes : List[str], tokenizer):
    raw_suffix_toks, attention_mask = tokenizer(suffixes, 
                                                    add_special_tokens=False, 
                                                    return_tensors="pt", 
                                                    padding = True).values() #remove start of sequence character
    good_idx = torch.ones(len(raw_suffix_toks), dtype=torch.bool)
    if torch.any(attention_mask == 0):
        printd("Attention mask has zeros")
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
    suffix_toks = process_suffix_toks(suffix_toks)
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
    
    with model.hooks(fwd_hooks, hooks_filter):
        if hooks_filter == []:
            logits = model(batch, past_kv_cache = kv_cache)
            return logits, None
        else:
            return model.run_with_cache(batch, past_kv_cache = kv_cache, names_filter = names_filter)
    
    
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
        logits = run_with_kv_cache(batch, kv_cache, model, fwd_hooks, hooks_filter)[:, -1].detach()
        probs = torch.softmax(logits, dim=-1)
        max_probs, max_tokens = torch.max(probs, dim=-1)
        
        all_probs.append(max_probs)
        all_toks.append(max_tokens)
        runner.update(len(batch))
        
    all_probs = torch.cat(all_probs, dim=0)
    all_toks = torch.cat(all_toks, dim=0)
    return all_probs, all_toks