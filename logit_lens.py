# %%

%load_ext autoreload
%autoreload 2
# %%

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_lens import HookedTransformer, utils
#!pip install git+https://github.com/callummcdougall/eindex.git
import eindex
import dq
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
# %%

model_name = "gpt2"
model = HookedTransformer.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# %%

df = pd.read_csv('data/test/german_story.txt')
german_story = [word for word in df['de']]
english_story = [word for word in df['en']]

# %%

@torch.no_grad
def logit_lens(nn_model, prompts, only_last_token=True):
    """
    Compute the logits for each layer of a neural network model given a set of prompts.

    Args:
        nn_model (torch.nn.Module): The neural network model.
        prompts (list[str]): The list of prompts.
        only_last_token (bool, optional): Whether to consider only the last token of each prompt. 
            Defaults to True.

    Returns:
        torch.Tensor: The logits per layer of the model.

    """
    model.eval()
    tok_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index for each prompt
    last_token_index = tok_prompts.attention_mask.cumsum(1).argmax(-1)
    
    output, cache = model.run_with_cache(prompts) #Expensive!
    
    hidden_l = []
    
    for i in range(model.cfg.n_layers):
        layer_cache = cache[f'blocks.{i}.hook_resid_post']  # (batch, seq, d_model)
        if only_last_token:
            layer_cache = eindex(layer_cache, last_token_index, "i [i] j") # (batch, d_model)
        hidden_l.append(layer_cache) # (batch, seq?, d_model)
            
    hidden = torch.stack(hidden_l, dim=1)  # (batch, num_layers, seq?, d_model)
    rms_out_ln = model.ln_final(hidden) # (batch, num_layers, seq?, d_model)
    logits_per_layer = model.unembed(rms_out_ln) # (batch, num_layers, seq?, vocab_size)
    
    return logits_per_layer
# %%




# %%
# prompt_zh_to_fr = "中文:花 français: fleur 中文:山 français: montagne 中文:月 français: lune 中文:水 français: eau"

# df = pd.read_csv('data/test/single_tok_lang6.csv')

# # %%
# def find_common_prefix(strings):
#     if not strings:
#         return ""
    
#     prefix = strings[0]
#     for string in strings[1:]:
#         while not string.startswith(prefix):
#             prefix = prefix[:-1]
#             if not prefix:
#                 return ""
    
#     return prefix


# @torch.no_grad
# def logit_lens_fast(nn_model, prompts):
#     """
#     Get the probabilities of the next token for the last token of each prompt at each layer using the logit lens.

#     Args:
#         nn_model: NNSight LanguageModel object
#         prompts: List of prompts or a single prompt
#         qk_caching: Perform a forward pass on the common prefix of the prompts to cache the query and key tensors.

#     Returns:
#         A tensor of shape (num_layers, num_prompts, vocab_size) containing the probabilities
#         of the next token for each prompt at each layer. Tensor is on the CPU.
#     """
    
#     prefix = find_common_prefix(prompts)
    
#     model.eval()
#     tok_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
#     # Todo?: This is a hacky way to get the last token index for each prompt
#     last_token_index = tok_prompts.attention_mask.cumsum(1).argmax(-1)
    
#     output, cache = model.run_with_cache(prompts) #Expensive!
    
#     hidden_l = []
#     for i in range(model.cfg.n_layers):
#         hidden_l.append(eindex(cache[f'blocks.{i}.hook_resid_post'], last_token_index, "i [i] j"))
    
#     hiddens = torch.stack(hidden_l, dim=0)  # (num_layers, num_prompts, d_model) 
#     rms_out_ln = model.ln_final(hiddens) # (num_layers, num_prompts, d_model)
#     logits_per_layer = model.unembed(rms_out_ln) # (num_layers, num_prompts, vocab_size)
#     probs_per_layer = logits.softmax(-1) # (num_layers, num_prompts, vocab_size)
    
    
#     assert torch.allclose(
#         logits_per_layer[-1],
#         eindex(output, last_token_index, "i [i] j")
#     )
#     return probs_per_layer

# # %%
# # Common prefix
# common_prefix = "The cat sat on the"

# # Unique suffixes
# suffixes = ["mat.", "rug.", "floor."]

# # Tokenize the common prefix
# tokens_prefix = tokenizer(common_prefix, return_tensors="pt")
# with torch.no_grad():
#     # Forward pass for the common prefix to obtain the last hidden state
#     outputs_prefix = model(**tokens_prefix, output_hidden_states=True)
#     hidden_states_prefix = outputs_prefix.hidden_states[-1][:, -1, :]

# # Process each suffix
# for suffix in suffixes:
#     # Tokenize suffix
#     tokens_suffix = tokenizer(suffix, return_tensors="pt")
#     # Concatenate the last hidden state of the prefix with the input ids of the suffix
#     # Adjust depending on whether you need to include the prefix's last token for context
#     inputs_with_state = {"input_ids": tokens_suffix["input_ids"], "past_key_values": outputs_prefix.past_key_values}
    
#     with torch.no_grad():
#         # Forward pass for the suffix, reusing the hidden state
#         outputs_suffix = model(**inputs_with_state)
    
#     # Convert logits to probabilities
#     probabilities = torch.softmax(outputs_suffix.logits[:, -1, :], dim=-1)
    
#     # Here you can extract and process the predictions as needed
#     print(f"Processed suffix '{suffix}'")