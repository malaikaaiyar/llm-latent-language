
#Joseph Miller code 

# minimal code
kv_cache = HookedTransformerKeyValueCache.init_cache(
    model.cfg, model.cfg.device, prefix_tokens.shape[0]
)
model(prefix_tokens, past_kv_cache=kv_cache)
kv_cache.freeze()
with_cache_logits = model(rest_of_tokens, past_kv_cache=kv_cache)

# integrated
import torch as t
from transformer_lens import HookedTransformer
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache

device = "cuda" if t.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained_no_processing("gpt2", device=device)
tokenizer = model.tokenizer

prefix = "the cat sat on the"
suffixes = [" mat quickly", " chair sneakily", " table slowly and carefully"]
n_suffix = len(suffixes)

prefix_toks = model.to_tokens(prefix, prepend_bos=True)
prefix_toks = prefix_toks.repeat(n_suffix, 1)
suffix_toks = model.to_tokens(suffixes, prepend_bos=False, padding_side="left")

kv_cache = HookedTransformerKeyValueCache.init_cache(
    model.cfg, model.cfg.device, n_suffix
)
_, prefix_cache = model.run_with_cache(prefix_toks, past_kv_cache=kv_cache)
kv_cache.freeze()
with model.hooks(fwd_hooks=fwd_hooks):
    output, cache = model.run_with_cache(suffix_toks, names_filter=hookpoint_names_filter, past_kv_cache=kv_cache)