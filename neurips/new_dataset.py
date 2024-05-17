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
# %%

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', use_fast=False, add_prefix_space=False)
vocab = tokenizer.get_vocab()
# %%
english_dict = pd.read_csv('data/dict/en-simple-3000.txt', sep='\t', header=None)
english_dict[1] = english_dict[0].apply(lambda x: '▁' + x)
english_dict = english_dict[(english_dict[0].isin(vocab)) | (english_dict[1].isin(vocab))]
english_dict['en_token'] = english_dict.apply((lambda row: row[1] if row[1] in vocab else row[0]), axis=1)
english_dict = english_dict.drop(columns=[1])
english_dict = english_dict.rename(columns={0: 'en'})
assert all([x in vocab for x in list(english_dict['en_token'])]), 'Not all tokens in english_dict are in vocab'
english_words = set(english_dict['en'])
# %%

zh_to_en = pd.read_csv('data/new/zh_to_en.csv')
melted = zh_to_en.melt(id_vars=['Chinese'], value_vars=['Synonym1', 'Synonym2', 'Synonym3', 'Synonym4'], 
                       var_name='SynonymType', value_name='Synonym')
# Drop rows with NaN values in 'Synonym'
melted = melted.dropna(subset=['Synonym'])
# Filter rows where the synonym is in english_words
valid_synonyms = melted[melted['Synonym'].isin(english_words)]
# Drop duplicates to keep the first valid synonym for each Chinese character
valid_synonyms = valid_synonyms.drop_duplicates(subset=['Chinese'], keep='first')
# Create the new DataFrame with the 'Chinese' column and the first matched synonym
filtered_zh_to_en = valid_synonyms[['Chinese', 'Synonym']]
filtered_zh_to_en = filtered_zh_to_en.rename(columns={'Chinese': 'zh', 'Synonym': 'Synonym'})
merged_dict = pd.merge(filtered_zh_to_en, english_dict, left_on='Synonym', right_on='en', how='inner')
merged_dict = merged_dict.drop(columns=['Synonym'])

merged_dict[['zh', 'en']].to_csv('merged_dict.csv', index=False)
# %%
# zh_vocab = {}
# for key, value in vocab.items():
#     if len(key) == 1 and is_chinese_char(key):
#         zh_vocab[key] = value

zh_en_fr = pd.read_csv('data/new/zh_en_fr.csv')
zh_en_fr_best = pd.read_csv('data/new/no_duplicates.csv')

# Step 1: Identify duplicates in the 'en' column of DataFrame A
duplicates_in_a = zh_en_fr[zh_en_fr.duplicated(subset='en', keep=False)]['en'].unique()
replacement_rows = zh_en_fr_best[zh_en_fr_best['en'].isin(duplicates_in_a)]
zh_en_fr_no_dup = zh_en_fr[~zh_en_fr['en'].isin(duplicates_in_a)]
zh_en_fr_no_dup = pd.concat([zh_en_fr_no_dup, replacement_rows], ignore_index=True)

zh_en_fr_no_dup.to_csv('data/new/zh_en_fr_no_dup.csv', index=False)
# %%





# zh_en_fr_data = pd.read_csv('data/new/zh_en_fr.csv')
# zh_en_fr_data = zh_en_fr_data.sort_values(by='en')
# duplicate_en = zh_en_fr_data[zh_en_fr_data.duplicated('en', keep=False)]
# duplicate_en.to_csv('data/new/duplicates.csv', index=False)



#zh_to_en_data = pd.merge(zh_to_en_data, merged_dict, on='en', how='inner')
# %%
zh_en_fr_de = pd.read_csv('data/new/zh_en_fr_de.csv')
# %%
filtered_rows2 = []
# Iterate over each row in the dataframe
for index, row in zh_en_fr_de.iterrows():
    zh_token = row['zh']
    en_token = row['en']
    fr_token = row['fr']
    de_token = row['de']
    

    # Check if any translations match each other
    if en_token == fr_token or en_token == de_token or fr_token == de_token:
        print(f"Warning: Translations for '{en_token} {fr_token} {de_token}' matched each other.")
        continue

    # Check if tokens exist in vocab
    zh_tok_str = zh_token if zh_token in vocab else None
    en_tok_str = f'▁{en_token}' if f'▁{en_token}' in vocab else None
    fr_tok_str = f'▁{fr_token}' if f'▁{fr_token}' in vocab else None
    de_tok_str = f'▁{de_token}' if f'▁{de_token}' in vocab else None

    # If all conditions are met, add the row to the filtered list
    if zh_tok_str and en_tok_str and fr_tok_str and de_tok_str:
        zh_tok_id = vocab[zh_tok_str]
        en_tok_id = vocab[en_tok_str]
        fr_tok_id = vocab[fr_tok_str]
        de_tok_id = vocab[de_tok_str]
        filtered_rows2.append({
            'zh': row['zh'],
            'en': row['en'],
            'fr': row['fr'],
            'de': row['de'],
            'zh_tok_str': zh_tok_str,
            'zh_tok_id': zh_tok_id,
            'en_tok_str': en_tok_str,
            'en_tok_id': en_tok_id,
            'fr_tok_str': fr_tok_str,
            'fr_tok_id': fr_tok_id,
            'de_tok_str': de_tok_str,
            'de_tok_id': de_tok_id
        })
    else:
        print(f"Warning: One or more tokens not found in vocab for '{en_token}'.")

# Create a new dataframe from the filtered rows
filtered_df2 = pd.DataFrame(filtered_rows2)
filtered_df2[['zh', 'en', 'fr', 'de']].to_csv('data/new/zh_en_fr_de_tokenizable.csv', index=False)
print(filtered_df2)
# %%

def test_tokenizability(all_langs, model):
    vocab = model.tokenizer.get_vocab()
    lang_codes = all_langs.columns
    lang_counts = {lang: 0 for lang in lang_codes}
    lang_counts['size'] = len(all_langs)
    for index, row in all_langs.iterrows():
        for lang in lang_codes:
            token = row[lang]
            if lang == 'zh':
                if token in vocab:
                    lang_counts[lang] += 1
                else:
                    print(f"Warning: Token '{token}' not found in vocab for '{lang}'.")
            else:
                if "▁" + token in vocab:
                    lang_counts[lang] += 1
                elif lang == 'en':
                    print(f"Warning: {list(row)} Token '{token}' not found in vocab for '{lang}'.")
    return lang_counts
    


all_langs = pd.read_csv('data/new/zh_en_fr.csv')

# %%
all_langs['zh']
# %%

all_langs = pd.read_csv('data/new/zh_CN_TW.csv')

lang_codes = ['zh', 'zh_CN', 'zh_TW']
lang_counts = {lang: 0 for lang in lang_codes}
for index, row in all_langs.iterrows():
    s, t = row['zh_CN'], row['zh_TW']
    if s != t and s in vocab and t in vocab:
        print(list(row))
print(lang_counts)
# %%
