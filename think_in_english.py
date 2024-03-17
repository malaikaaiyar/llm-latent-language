# %%
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass, field
torch.set_grad_enabled(False)
import warnings

from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer

from dq_utils import tok_to_id, get_space_char, print_tok, raw_tok_to_id, plot_ci

@dataclass
class Config:
    source_lang: str = 'zh'
    target_lang: str = 'fr'
    think_lang: str = 'en'
    model_name: str = 'meta-llama/Llama-2-7b-hf'
    model_kwargs: dict = field(default_factory=dict)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cfg = Config()
cfg.model_kwargs = {'use_fast': False, 'add_prefix_space': False}
# # Set torch device to use CPU only
# device = torch.device('cpu')
# tokenizer = HookedTransformer.from_pretrained(cfg.model_name, device=device).tokenizer

# # Replace 'llama-2-model-name' with the actual model name for Llama-2
# tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **cfg.model_kwargs)

tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", use_fast=False, add_prefix_space=False)

tokenizer_mistral = AutoTokenizer.from_pretrained('mistralai/Mistral-7B-v0.1', 
                                                  add_prefix_space=False, 
                                                  use_fast=False,
                                                  legacy=False)
# # mistral also adds leading spaces https://github.com/huggingface/transformers/issues/29452 use legacy=False

tokenizer_qwen = AutoTokenizer.from_pretrained('Qwen/Qwen-7B', trust_remote_code=True)
# # qwen tokenizes into binary?

tokenizer_tinystories = AutoTokenizer.from_pretrained('roneneldan/TinyStories-33M', trust_remote_code=True)
tokenizer_gpt2 = AutoTokenizer.from_pretrained('gpt2', trust_remote_code=True, use_fast=False)

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **cfg.model_kwargs)
# with open('./vocabs/tinystories-33M-vocab.txt', 'w') as f:
#     for token, index in tokenizer_tinystories.get_vocab().items():
#         f.write(f"{token}\t{index}\n")


# Replace 'mistral-7b-model-name' with the actual model name for Mistral-7B
# tokenizer = AutoTokenizer.from_pretrained('gpt2', 
#                                             # use_fast=False)


#decode some chinese
#assert 29871 not in tokenizer.encode("你好"), "Use add_prefix_space=False for Chinese tokenization to avoid SPIECE_UNDERLINE character."
# see https://github.com/huggingface/transformers/issues/26273
# we want the space to be added when tokenizing an entire sequence, but not a small sequence to be concatenated
# put the space manually 

# Note by default the tokenizer will add a space before the first token, which is not what we want for Chinese
# i.e. "hello world" converts to " hello world" and tokenizes as ["<s>", "▁hello", "▁world"] 
#this may mess up tokenization to deliberately not do this.

# %%







# %%


        # "ko" : "한국어",  
        
#"中文: 你 français: tu 中文: 月 français: lune 中文: 他 français: il 中文: 她 français: elle"
        
# %%

# with open('./vocabs/gpt2-vocab.txt', 'w') as f:
#     for token, index in tokenizer.get_vocab().items():
#         f.write(f"{token}\t{index}\n")

# %%

# %%
# %%
def gen_prompts(word_dict,cfg, tokenizer):
    base_prompt = ""    
    
    # Take the first 4 words from the word_dict
    base_words, complete_words = list(word_dict.keys())[:4], list(word_dict.keys())[4:]
    
    prompts = []
    answers = []
    latents = []
    
    src_lang = lang2name[cfg.source_lang]
    dest_lang = lang2name[cfg.target_lang]
    space_char = get_space_char(tokenizer)
    for word in base_words:
        src_word, _ = word_dict[word][cfg.source_lang]
        src_word = src_word.replace(space_char, "")
        
        dest_word, _ = word_dict[word][cfg.target_lang]
        dest_word = dest_word.replace(space_char, "")
        
        prompt = f"{src_lang}: {src_word} {dest_lang}: {dest_word} "
        base_prompt += prompt
 
    print(base_prompt)
    
    for word in complete_words:
        
        src_word, _ = word_dict[word][cfg.source_lang]
        src_word = src_word.replace(space_char, "")
        
        dest_word, _ = word_dict[word][cfg.target_lang]
        dest_word = dest_word.replace(space_char, "")
        
        prompt = f"{src_lang}: {src_word} {dest_lang}: "
        answer, answer_tok = word_dict[word][cfg.target_lang]
        latent, latent_tok = word_dict[word][cfg.think_lang]
        prompts.append(base_prompt + prompt)
        answers.append((answer, answer_tok))
        latents.append((latent, latent_tok))
        
    return prompts, answers, latents

prompts, answers, latents = gen_prompts(filtered_word_dict, cfg, tokenizer)

# for word, translations in word_dict.items():
#     for lang, translation in translations.items():
#         assert translation in tokenizer.get_vocab() or '▁'+translation in tokenizer.get_vocab(), f"Missing {lang} token for {translation}"


#prompts, answers, think = gen_prompts(word_dict, cfg)
# %%
# def gen_tokens(prompts, answers, think, tokenizer):
#     tokenized_prompts = [tokenizer.encode(prompt, return_tensors="pt") for prompt in prompts]
#     answer_ids = [tokenizer.encode(answer, return_tensors="pt", add_special_tokens=False) for answer in answers]
#     latent_ids = [tokenizer.get_vocab()[think] for think in think]
#     return tokenized_prompts, answer_ids, latent_ids

# tokenized_prompts, answer_ids, latent_ids = gen_tokens(prompts, answers, think, tokenizer)

# %%


# %%


# Set torch device to use CPU only
if 'model' not in locals():
    device = torch.device('cpu')
    model = HookedTransformer.from_pretrained(cfg.model_name, device=device)

    device = cfg.device
    model.to(device)

# # Set torch device to use CPU only
# device = torch.device('cpu')
# tokenizer = HookedTransformer.from_pretrained(cfg.model_name, device=device).tokenizer


latent_probs = torch.zeros(model.cfg.n_layers, len(prompts))
dest_probs = torch.zeros(model.cfg.n_layers, len(prompts))

for i, prompt in tqdm(enumerate(prompts)):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output, cache = model.run_with_cache(tokens)
    
    for j in range(model.cfg.n_layers):
        resid = cache[f'blocks.{j}.hook_resid_post'] 
        ln_resid = model.ln_final(resid)
        logits = model.unembed(ln_resid)
        #logits = ln_resid[0, -1, :] @ model.unembed.W_U + model.unembed.b_U
        latent_prob = torch.softmax(logits[0, -1, :], dim=-1)[latents[i][1]].item()
        dest_prob = torch.softmax(logits[0, -1, :], dim=-1)[answers[i][1]].item()
        
        latent_probs[j, i] = latent_prob
        dest_probs[j, i] = dest_prob

# %%


plot_ci(latent_probs, cfg.think_lang)
plot_ci(dest_probs, cfg.target_lang)
plt.legend()
plt.show()
# %%