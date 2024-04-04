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
import json
from dq import lang2name

@dataclass
class Config:
    source_lang: str = 'zh'
    target_lang: str = 'ko'
    think_lang: str = 'en'
    model_name: str = 'meta-llama/Llama-2-7b-hf'
    model_kwargs: dict = field(default_factory=dict)
    word_dict_path: str = "data/filtered_word_dict.json"
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cfg = Config()
cfg.model_kwargs = {'use_fast': False, 'add_prefix_space': False}
# %%

# # Set torch device to use CPU only
# device = torch.device('cpu')
# tokenizer = HookedTransformer.from_pretrained(cfg.model_name, device=device).tokenizer

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_prefix_space=False)
model = HookedTransformer.from_pretrained_no_processing(model_name, 
                                                        dtype=torch.float16, 
                                                        device='cuda:0', 
                                                        low_cpu_mem_usage=True)


tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **cfg.model_kwargs)


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

# %%
with open(cfg.word_dict_path, 'r') as f:
    filtered_word_dict = json.load(f)

prompts, answers, latents = gen_prompts(filtered_word_dict, cfg, tokenizer)
# %%

latent_probs = torch.zeros(model.cfg.n_layers, len(prompts))
dest_probs = torch.zeros(model.cfg.n_layers, len(prompts))

latent_probs_steer = torch.zeros(model.cfg.n_layers, len(prompts))
dest_probs_steer = torch.zeros(model.cfg.n_layers, len(prompts))


W_U, b_U = model.unembed.W_U, model.unembed.b_U
for i, prompt in tqdm(enumerate(prompts)):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    output, cache = model.run_with_cache(tokens)
    
    for j in range(model.cfg.n_layers):
        resid = cache[f'blocks.{j}.hook_resid_post'][:, -1, :] 
        ln_resid = model.ln_final(resid)
        #logits = model.unembed(ln_resid)
        
        alpha = (ln_resid @ W_U.T[latents[i][1]]) / (W_U.T[latents[i][1]] @ W_U.T[latents[i][1]])
        ln_resid_steer = ln_resid - alpha * W_U.T[latents[i][1]] 
        
        logits = ln_resid @ model.unembed.W_U + model.unembed.b_U
        logit_steer = ln_resid_steer @ model.unembed.W_U + model.unembed.b_U
        
        latent_prob_steer = torch.softmax(logit_steer[0, :], dim=-1)[latents[i][1]].item()
        dest_prob_steer = torch.softmax(logit_steer[0, :], dim=-1)[answers[i][1]].item()
        
        latent_prob = torch.softmax(logits[0, :], dim=-1)[latents[i][1]].item()
        dest_prob = torch.softmax(logits[0, :], dim=-1)[answers[i][1]].item()
        
        latent_probs[j, i] = latent_prob
        dest_probs[j, i] = dest_prob
        
        latent_probs_steer[j, i] = latent_prob_steer
        dest_probs_steer[j, i] = dest_prob_steer
        
        
# %%
latent_probs_steer = torch.zeros(model.cfg.n_layers, len(prompts))
dest_probs_steer = torch.zeros(model.cfg.n_layers, len(prompts))

from torch import Tensor
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookPoint

def hook_function(
    tblock_output: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    latent_tok_id: Int,
) -> Float[Tensor, "batch seq dmodel"]:
    W_U = model.unembed.W_U
    # modify attn_pattern (can be inplace)
    
    last_tblock = tblock_output[:, -1]
    
    latent_tok_id = latent_tok_id + 1
    
    alpha = (last_tblock @ W_U.T[latent_tok_id]) / (W_U.T[latent_tok_id] @ W_U.T[latent_tok_id])
    last_tblock_steer = last_tblock - alpha * W_U.T[latent_tok_id]
    tblock_output[:, -1] = last_tblock_steer
    return tblock_output


W_U, b_U = model.unembed.W_U, model.unembed.b_U
for i, prompt in tqdm(enumerate(prompts)):
    tokens = tokenizer.encode(prompt, return_tensors="pt").to(device)
    
    temp_hook_fn = lambda tblock_output, hook: hook_function(tblock_output, hook, latents[i][1])
    with model.hooks(fwd_hooks=[(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in range(model.cfg.n_layers)]):
        output, cache = model.run_with_cache(tokens)
    
    for j in range(model.cfg.n_layers):
        resid = cache[f'blocks.{j}.hook_resid_post'][:, -1, :] 
        ln_resid = model.ln_final(resid)
        #logits = model.unembed(ln_resid)
       
        logits = ln_resid @ model.unembed.W_U + model.unembed.b_U
        
        latent_prob = torch.softmax(logits[0, :], dim=-1)[latents[i][1]].item()
        dest_prob = torch.softmax(logits[0, :], dim=-1)[answers[i][1]].item()

        latent_probs_steer[j, i] = latent_prob
        dest_probs_steer[j, i] = dest_prob

# %%
from matplotlib import pyplot as plt
fig, ax = plt.subplots()
plot_ci(latent_probs, ax,label=cfg.think_lang)
plot_ci(dest_probs, ax, label = cfg.target_lang)
plt.legend()
plt.title(f"Language Probability per Layer {cfg.model_name}")
plt.xlabel('Layer')
plt.ylabel('Probability')
plt.show()
# %%
fig, ax = plt.subplots()
plot_ci(latent_probs_steer, ax,label=cfg.think_lang)
plot_ci(dest_probs_steer, ax, label = "ko interv")
plot_ci(dest_probs, ax, label = cfg.target_lang)

plt.legend()
plt.title(f"Language Probability per Layer {cfg.model_name}")
plt.xlabel('Layer')
plt.ylabel('Probability')
plt.show()




# %%
avg_wU = model.unembed.W_U.mean(dim=0)
sorted_avg_wU = torch.sort(avg_wU, descending=True).values
plt.plot(sorted_avg_wU.cpu())
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('Sorted Average W_U on Log Scale')
plt.show()
# %%
