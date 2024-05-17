# %%
%load_ext autoreload
%autoreload 2

# %%
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm import tqdm

# === Typing Libraries ===
from typing import Tuple, List, Optional, Dict, Callable, Iterable, Any
from jaxtyping import Int, Float
from beartype import beartype

# ==== Torch/Transformer Libraries ====
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer

# ==== Custom Libraries ====
import gen_data
from utils import plot_ci_plus_heatmap
from tuned_lens_wrap import load_tuned_lens
from dq_utils import proj, entropy, plot_ci, is_chinese_char
from logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = HookedTransformer.from_pretrained_no_processing('meta-llama/Llama-2-7b-hf',
                                                            device=device, 
                                                            dtype=torch.float16)

# %%
english_dict = pd.read_csv('data/dict/en-simple-3000.txt', sep='\t', header=None)


# %%
all_langs = pd.read_csv('data/new/zh_en_fr.csv')

# Check if every entry under 'en' is in the simple dictionary
for en_word in all_langs['en']:
    if en_word not in english_dict.values:
        print(f"Word '{en_word}' not found in the simple dictionary.")
# %%
@torch.no_grad
def llama_translate(src_word, model):
    model.eval()
    if len(src_word) == 1 and is_chinese_char(src_word):
        src_lang = "zh"
    else:
        src_lang = "en"
    vocab = model.tokenizer.get_vocab()
    assert src_word in vocab or "▁" + src_word in vocab, f"Input string {src_word} not in vocabulary"
    if src_lang == "zh":
        prompt = f"中文: 水 English: water\n中文: 中 English: middle\n中文: 三 English: three\n中文: 女 English: woman\n中文: {src_word} English:"
    elif src_lang == "en":
        prompt = f"English: water 中文: 水\nEnglish: middle 中文: 中\nEnglish: three 中文: 三\nEnglish: woman 中文: 女\nEnglish: {src_word} 中文: "
    else:
        raise NotImplementedError(f"Language {src_lang} not implemented")
    tokens = model.tokenizer.encode(prompt, return_tensors="pt")
    logits = model(tokens)[0, -1].detach().cpu()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_tok = torch.topk(probs, 5)
    top_tokens = model.tokenizer.convert_ids_to_tokens(top_tok.tolist())
    #df = pd.DataFrame({'Token': top_tokens, 'Probability': top_probs.tolist()})
    torch.cuda.empty_cache()
    return top_tokens, top_probs

# %%
from tqdm import tqdm
import pickle
llama_translations = []
for row in tqdm(all_langs.iterrows(), total=len(all_langs)):
    zh, en = row[1]['zh'], row[1]['en']
    zh_to_en_tok, zh_to_en_prob = llama_translate(zh, model)
    top_en = zh_to_en_tok[0]
    if top_en[0] != "▁":
        #print(f"Expected: {zh}, Got: {top_en} prob {zh_to_en_prob[0]}")
        llama_translations.append({'zh' : zh, 'en' : en, 'en_T' : top_en, 'en_T_all' : zh_to_en_tok, 'en_T_prob' : zh_to_en_prob,
                                   'zh_rec' : None, 'zh_rec_prob' : None, 'zh_rec_true' : None, 'zh_rec_prob_true' : None})
        continue
    
    if top_en[1:] != en:
        #print(f"Expected: {en}, Got: {top_en} prob {zh_to_en_prob[0]}")
        en_to_zh_tok_alt, en_to_zh_prob_alt = llama_translate(top_en[1:], model)
        en_to_zh_tok, en_to_zh_prob = llama_translate(en, model)
    else:
        en_to_zh_tok, en_to_zh_prob = llama_translate(en, model)
        en_to_zh_prob_alt, en_to_zh_tok_alt = en_to_zh_prob, en_to_zh_tok
    
    top_zh = en_to_zh_tok[0]
    top_zh_alt = en_to_zh_tok_alt[0]
    trans = {'zh' : zh,
             'en' : en,
             'en_T' : top_en,
             'en_T_all' : en_to_zh_tok,
             'en_T_prob' : zh_to_en_prob,
             'zh_rec' : top_zh,
             'zh_rec_prob' : en_to_zh_prob,
             'zh_rec_true' : top_zh_alt,
             'zh_rec_prob_true' : en_to_zh_prob_alt}
    llama_translations.append(trans)
# %%
# Pickle llama_translations
import pickle
pickle.dump(llama_translations, open('data/new/llama_translations.pkl', 'wb'))
# %%
llama_filter = []
for x in llama_translations:
    if x['zh'] == x['zh_rec_true'] and x['en_T_prob'][0] > 0.5 and x['zh_rec_prob_true'][0] > 0.5:
        llama_filter.append(x)

en_probs = torch.Tensor([x['en_T_prob'][0] for x in llama_filter])
zh_probs = torch.Tensor([x['zh_rec_prob_true'][0] for x in llama_filter])
min_prob = torch.minimum(en_probs, zh_probs)
_, idx = torch.sort(min_prob)

plt.plot(en_probs[idx], label = "en")
plt.plot(zh_probs[idx], label = 'zh')
plt.legend()
plt.show()
# %%
print("zh,en")
for x in llama_filter:
    zh, en = x['zh'], x['en_T']
    print(f"{zh},{en}")


# %%
for row in tqdm(all_langs.iterrows()):
    zh, en = row[1]['zh'], row[1]['en']
    zh_to_en_tok, zh_to_en_prob = llama_translate(zh, model)
    en_to_zh_tok, en_to_zh_prob = llama_translate(en, model)
    if zh == en_to_zh_tok[0]:
        continue
    print(zh, en, zh_to_en_tok, zh_to_en_prob, en_to_zh_tok, en_to_zh_prob)
    break
    # if zh_to_en_tok[0] == "▁" and zh_to_en_tok[1:] == en and en_to_zh_tok[0] == zh:
    #     continue
    # else:
    #     print
# %%