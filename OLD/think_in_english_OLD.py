# %%
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
torch.set_grad_enabled(False)
from transformers import AutoTokenizer
from utils.misc import plot_ci, lang2name
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass, field
# %%
# Set torch device to use CPU only



@dataclass
class Config:
    source_lang: str = 'zh'
    target_lang: str = 'ko'
    think_lang: str = 'en'
    model_name: str = 'meta-llama/Llama-2-7b-hf'
    word_dict_path: str = 'data/filtered_word_dict.json'
    base_prompt: str = '中文:花 한국어:꽃 中文:山 한국어:산 中文:月 한국어:달 中文:水 한국어:물 '
    model_kwargs: dict = field(default_factory=dict)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cfg = Config()
cfg.model_kwargs = {'use_fast': False, 'add_prefix_space': False}


device = torch.device('cpu')

if 'model' not in locals():
    model = HookedTransformer.from_pretrained(cfg.model_name, device=device)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **cfg.model_kwargs)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


with open(cfg.word_dict_path, 'r') as f:
    word_dict = json.load(f)

#base_prompt = '中文:花 한국어:꽃 中文:山 한국어:산 中文:月 한국어:달 中文:水 한국어:물 '
#base_prompt = '中文:"花" 中文:"花" 中文:"山" 中文:"山" 中文:"月" 中文:"月" 中文:"水" 中文:"水" '
#base_prompt = '花花山山月月水水'

# %%

# %%

intermediate_probs = torch.zeros((4, model.cfg.n_layers, len(word_dict)))
entropy = torch.zeros((model.cfg.n_layers, len(word_dict)))
prompts = []
src_lang = lang2name[cfg.source_lang]
dest_lang = lang2name[cfg.target_lang]

for i, baseword in tqdm(enumerate(word_dict.keys())):
    src_word, src_id = word_dict[baseword][cfg.source_lang]
    dest_word, dest_id = word_dict[baseword][cfg.target_lang]
    think_word, think_id = word_dict[baseword][cfg.think_lang]
    
    lang_ids = [src_id, dest_id, think_id]
    
    
    suffix_prompt = f'{src_lang}:{src_word} {dest_lang}:'
    prompt = cfg.base_prompt + suffix_prompt
    prompts.append(prompt)
    output, cache = model.run_with_cache(prompt) #EXPENSIVE PART!
    
    for j in range(model.cfg.n_layers):
        resid = cache[f'blocks.{j}.hook_resid_post'] 
        ln_resid = model.ln_final(resid)
        logits = model.unembed(ln_resid)
        probs = torch.softmax(logits[0, -1, :], dim=-1)
        #logits = ln_resid[0, -1, :] @ model.unembed.W_U + model.unembed.b_U
        
        for k in range(len(lang_ids)):
            intermediate_probs[k, j, i] = probs[lang_ids[k]].item()
            entropy[j,i] = -torch.sum(probs * torch.log2(probs))
intermediate_probs[3] = 1 - intermediate_probs[:3].sum(dim=0)

# %%

from matplotlib.font_manager import FontProperties
import logging
import numpy as np
font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')  # Specify the path to your font file
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

labels = ["中文", "한국어", "▁en", "other"]
for i, label in enumerate(labels):
    plot_ci(intermediate_probs[i], ax1, label = label)

#plot_ci(quote_probs, ':', ax)
ax1.legend(loc='upper left', prop=font)
ax1.set_title(f'Layer-wise probabilities for zh -> ko translation', fontproperties=font)
ax1.set_xlabel('Layer')
ax1.set_ylabel('Probability')

ax2.plot(entropy.mean(dim=1))
ax2.set_title('Entropy of layer-wise probabilities', fontproperties=font)
ax2.set_xlabel('Layer')
ax2.set_ylabel('Entropy')
plt.show()



# %%



# %%

import matplotlib.pyplot as plt
import seaborn as sns
import torch






# You'll need to provide `cache`, `model`, and `tokenizer` when calling this function
example_prompt = cfg.base_prompt + '中文:音 한국어:' #음
fig, ax = plt.subplots(1, 1, figsize=(8, 16))
logit_lens(example_prompt, model, tokenizer, ax)
ax.set_title(f'Logit Lens prompt={example_prompt}', fontproperties=font)
ax.set_xlabel('Tokens')
ax.set_ylabel('Layers')
plt.show()
# output, cache = model.run_with_cache("The cat sat on the")
# k=10
# heatmap = torch.zeros(model.cfg.n_layers,k)
# plt.figure(figsize=(8, 16))
# for j in range(model.cfg.n_layers):
#     resid = cache[f'blocks.{j}.hook_resid_post']
#     ln_resid = model.ln_final(resid)
#     logits = model.unembed(ln_resid)
#     probs = torch.softmax(logits[0, -1, :], dim=-1).cpu()
#     top_probs, top_tok = torch.topk(probs, k)
#     heatmap[j] = top_probs
#     # Get the token names
#     token_names = tokenizer.convert_ids_to_tokens(top_tok)
#     for i, token_name in enumerate(token_names):
#         # Draw each token name; adjust y position based on `i` to avoid overlap
#         plt.text(i, j, token_name, fontsize=8, ha='center', va='center', rotation=45, color='white')
# %%
