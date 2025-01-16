# %%

'''
# use run_with_shared_prefix on prefix_toks : Int[Tensor, "seq"], 
                        #    suffix_toks : Int[Tensor, "batch seq2"], 
                        #    model, 
                        #    fwd_hooks = []):

# def gen_prompt_repeats(src_words = None, 
            #    src_lang = None, 
            #    num_examples= None):


def gen_common_suffixes_repeats(src_words, src_lang):


class Intervention:
    def __init__(self, func_name: str, layers: Iterable[int]):
    
hook_batch_reject

'''



from src.prompt import gen_prompt_repeats, gen_common_suffixes_repeats
import pandas as pd
from transformer_lens import HookedTransformer
from src.constants import LANG2NAME
from src.logit_lens import run_with_shared_prefix
from src.intervention import Intervention
import torch

# %%

df = pd.read_csv("data/butanium_v2.tsv", delimiter = '\t')
num_examples = 5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


if 'LOAD_MODEL' not in globals():
    model = HookedTransformer.from_pretrained_no_processing("meta-llama/Llama-2-7b-hf",
                                                            device=device,
                                                            dtype = torch.float16)
    tokenizer = model.tokenizer
    tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore
    LOAD_MODEL = False

# %%

latent_lang = 'en'

for src_lang in ['en', 'zh', 'fr', 'es', 'de', 'ru']:
    src_words = df[src_lang]
    latent_words = df[latent_lang]
    
    prefix = gen_prompt_repeats(src_words, src_lang, num_examples)
    suffixes = gen_common_suffixes_repeats(src_words, src_lang)
    
    # Create prefix tokens - shape [seq]
    prefix_toks = model.tokenizer(prefix, return_tensors='pt').input_ids[0].to(device)
    
    # Create suffix tokens with padding to ensure consistent sequence length
    suffix_tensors = [model.tokenizer(suffix, return_tensors='pt').input_ids[0].to(device) for suffix in suffixes]
    max_len = max(len(t) for t in suffix_tensors)
    
    padded_suffixes = []
    for t in suffix_tensors:
        if len(t) < max_len:
            padding = torch.full((max_len - len(t),), model.tokenizer.pad_token_id, dtype=t.dtype, device=device)
            padded_suffixes.append(torch.cat([t, padding]))
        else:
            padded_suffixes.append(t)
    suffix_toks = torch.stack(padded_suffixes)  # Already on device from earlier
    
    # Create indices for the last non-padding token in each sequence
    attention_mask = (suffix_toks != model.tokenizer.pad_token_id).long()
    suffix_indices = attention_mask.sum(dim=1) - 1  # Get index of last non-padding token
    
    # Create rejection subspace from the source words
    # src_ids = torch.tensor([model.tokenizer.encode(word)[0] for word in src_words], dtype=torch.long, device=device)
    latent_ids = torch.tensor([model.tokenizer.encode(word)[0] for word in latent_words], dtype=torch.long, device=device)
    batch_size = len(suffix_toks)
    rejection_subspace = model.unembed.W_U.T[latent_ids]  # Shape: [num_words, d_model]
    rejection_subspace = rejection_subspace.unsqueeze(0).expand(batch_size, -1, -1)  # Shape: [batch, num_words, d_model]
    
    # Create intervention with rejection subspace
    intervention = Intervention('hook_batch_reject', range(model.cfg.n_layers))
    fwd_hooks = intervention.fwd_hooks(model, 
                                     rejection_subspaces=rejection_subspace,
                                     latent_idx=latent_ids,
                                     suffix_idx=suffix_indices)  # Pass indices of last tokens

    output, cache = run_with_shared_prefix(prefix_toks, suffix_toks, model, fwd_hooks)
    
    # Get logits for the last token of each sequence
    logits = output[torch.arange(len(suffix_toks)), suffix_indices]  # [batch, vocab]
    probs = torch.softmax(logits, dim=-1)
    
    # Zero out unk token probability
    probs[:, model.tokenizer.unk_token_id] = 0
    
    # Get probabilities for the expected completions (the source words)
    src_word_probs = probs[:, src_ids]  # [batch, num_src_words]
    
    # Print results
    print(f"\n=== {src_lang} ===")
    print("Top 5 completions for each prompt:")
    for i, (suffix, prob_dist) in enumerate(zip(suffixes, probs)):
        top_tokens = torch.topk(prob_dist, k=5)
        top_probs = top_tokens.values
        top_ids = top_tokens.indices
        top_words = [model.tokenizer.decode(idx.item()) for idx in top_ids]
        
        print(f"\nPrompt: {suffix}")
        print("Expected completion:", src_words.iloc[i])
        print("Top completions:")
        for prob, word in zip(top_probs, top_words):
            print(f"{word}: {prob:.3f}")

    

# %%