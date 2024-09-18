# %%
# %load_ext autoreload
# %autoreload 2

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
import OLD_llama.gen_data as gen_data
from utils_plot import plot_ci_plus_heatmap
from tuned_lens_wrap import load_tuned_lens
from utils.misc import proj, entropy, plot_ci, is_chinese_char
from src.logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = HookedTransformer.from_pretrained_no_processing('meta-llama/Llama-2-7b-hf',
                                                            device=device, 
                                                            dtype=torch.float16)

# %%
torch.set_grad_enabled(False)
# %%
#english_dict = pd.read_csv('data/dict/en-simple-3000.txt', sep='\t', header=None)
#english_words = set(english_dict[0])
vocab = model.tokenizer.get_vocab()

zh_tokens = []
for key in vocab:
    if len(key) == 1 and is_chinese_char(key):
        zh_tokens.append(key)

# %%
@torch.no_grad()
def llama_translate(src_word, model, prompt_func):
    model.eval()
    vocab = model.tokenizer.get_vocab()
    assert src_word in vocab or "▁" + src_word in vocab, f"Input string {src_word} not in vocabulary"
    prompt = prompt_func(src_word)
    tokens = model.tokenizer.encode(prompt, return_tensors="pt")
    logits = model(tokens)[0, -1].detach().cpu()
    probs = torch.softmax(logits, dim=-1)
    top_probs, top_tok = torch.topk(probs, 5)
    top_tokens = model.tokenizer.convert_ids_to_tokens(top_tok.tolist())
    #df = pd.DataFrame({'Token': top_tokens, 'Probability': top_probs.tolist()})
    return top_tokens, top_probs

translation_bank = [
    {'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},
    'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'муж'},
    'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'три'},
    'new': {'zh': '新', 'en': 'village', 'fr': 'nouveau', 'de': 'neu', 'ru': 'пя'}},
    
    {'water': {'zh': '水', 'en': 'water', 'fr': 'eau', 'de': 'Wasser', 'ru': 'вода'},
    'middle': {'zh': '中', 'en': 'middle', 'fr': 'milieu', 'de': 'Mitte', 'ru': 'середина'},
    'three': {'zh': '三', 'en': 'three', 'fr': 'trois', 'de': 'drei', 'ru': 'три'},
    'woman': {'zh': '女', 'en': 'woman', 'fr': 'femme', 'de': 'Frau', 'ru': 'женщина'}}
]


def generate_translation_prompt(src_lang, dest_lang, word, prompt_bank = 0):
    if word[0] == "▁":
        word = word[1:]
        
    translations = translation_bank[prompt_bank]
    
    language_labels = {
        'zh': '中文',
        'en': 'English',
        'fr': 'Français',
        'de': 'Deutsch',
        'ru': 'Русский'
    }

    prompt = ""
    for key, translation in translations.items():
        prompt += f"{language_labels[src_lang]}: {translation[src_lang]} {language_labels[dest_lang]}: {translation[dest_lang]}\n"
    
    prompt += f"{language_labels[src_lang]}: {word} {language_labels[dest_lang]}:"
    
    # Ensure prompt ends with a space for Chinese
    if dest_lang == 'zh':
        prompt += ' '
    
    return prompt
# %%
@torch.no_grad()
def translate(src_words, src_lang, dest_lang, model = model, prompt_bank = 0):
    vocab = model.tokenizer.get_vocab()
    src_words_out = []
    dest_words = []
    dest_probs = []
    rev_probs = []
    correct = 0
    runner = tqdm(src_words, total=len(src_words))
    for idx, src_word in enumerate(runner):
        assert src_word in vocab or ("▁" + src_word in vocab), f"Input string {src_word} not in vocabulary"
        prompt = generate_translation_prompt(src_lang, dest_lang, src_word, prompt_bank)
        tokens = model.tokenizer.encode(prompt, return_tensors="pt")
        logits = model(tokens)[0, -1]
        probs = torch.softmax(logits, dim=-1).detach().cpu()
        dest_prob, dst_tok = torch.max(probs, dim=-1)
        dst_word = model.tokenizer.convert_ids_to_tokens(dst_tok.item())
        # if dst_word[0] != "▁" or dst_word[1:] not in english_words:
        #     runner.set_description(f"Failed: {correct} / {idx+1} {src_word} -> {dst_word}")
        #     #print(f"Translation {src_word} -> {dst_word} not in english dictionary")
        #     continue
        
        #Check we can translate back
        
        prompt_rev = generate_translation_prompt(dest_lang, src_lang, dst_word, prompt_bank)
        tokens = model.tokenizer.encode(prompt_rev, return_tensors = 'pt')
        logits = model(tokens)[0, -1]
        probs = torch.softmax(logits, dim=-1).detach().cpu()
        src_prob_rev, src_tok_rev = torch.max(probs, dim=-1)
        src_word_rev = model.tokenizer.convert_ids_to_tokens(src_tok_rev.item())
        if (src_word_rev != src_word and src_word_rev[1:] != src_word):
            runner.set_description(f"Failed: {correct} / {idx+1} {src_word} -> {dst_word} -> {src_word_rev}")
            #print(f"Translation {src_word} -> {dst_word} -> {src_word_rev}")
            continue
        
        dest_words.append(dst_word)
        dest_probs.append(dest_prob)
        rev_probs.append(src_prob_rev)
        src_words_out.append(src_word)
        correct += 1
        runner.set_description(f"Passed: {correct} / {idx+1} {src_word} -> {dst_word} -> {src_word_rev}")
        
    dest_probs = torch.Tensor(dest_probs)
    rev_probs = torch.Tensor(rev_probs)
    
    data = {
        src_lang: src_words_out,
        dest_lang: dest_words,
        f'{src_lang}_to_{dest_lang}_prob': dest_probs,
        f'{dest_lang}_to_{src_lang}_prob': rev_probs
    }

    df = pd.DataFrame(data)
    return df
# %%



bank = {"zf_to_en" : [],
            "en_to_fr" : [],
            "en_to_de" : [],
            "en_to_ru" : [],
            "all" : [],
            "filtered" : []}

for prompt_bank in range(len(translation_bank)):
    print(f"Prompt Bank: {prompt_bank}")

    llama_zf_to_en = translate(zh_tokens, 'zh', 'en', prompt_bank = prompt_bank)
    #llama_zf_to_en.rename(columns={'src_word': 'zh', 'dest_word': 'en', 'dest_prob': 'zh_to_en_prob', 'rev_prob': 'en_to_zh_prob'}, inplace=True)
    llama_en_to_fr = translate(list(llama_zf_to_en['en']), 'en', 'fr', prompt_bank = prompt_bank)
    llama_en_to_de = translate(list(llama_zf_to_en['en']), 'en', 'de', prompt_bank = prompt_bank)
    llama_en_to_ru = translate(list(llama_zf_to_en['en']), 'en', 'ru', prompt_bank = prompt_bank)

    llama_all = pd.merge(llama_zf_to_en, llama_en_to_fr, on='en', how='inner')
    llama_all = pd.merge(llama_all, llama_en_to_de, on='en', how='inner')
    llama_all = pd.merge(llama_all, llama_en_to_ru, on='en', how='inner')

    # Save filtered_df dataframe
    
    tolerance = 0.3
    mask = llama_all.filter(like='_prob').gt(tolerance).all(axis=1)
    # Filter the DataFrame using the mask
    filtered_df = llama_all[mask]
    filtered_df.reset_index(drop=True, inplace=True)
    #filtered_df.to_csv(os.path.join(save_dir, 'llama2_filtered_30_tol.csv'), index=False)
    
    
    bank["zh_to_en"].append(llama_zf_to_en)
    bank["en_to_fr"].append(llama_en_to_fr)
    bank["en_to_de"].append(llama_en_to_de)
    bank["en_to_ru"].append(llama_en_to_ru)
    bank["all"].append(llama_all)
    bank["filtered"].append(filtered_df)

# %%
# llama_all_old = pd.read_csv('data/new/llama_2.csv')
# llama_all_old = llama_all_old.loc[:, ~llama_all_old.columns.str.endswith('_y')]
# llama_all_old.rename(columns={col: col.split('_x')[0] for col in llama_all_old.columns if '_x' in col}, inplace=True)
# %%
lang_codes = ['zh', 'fr', 'de', 'ru']

def combine_dataframes(df1, df2):

# Perform an outer join on 'en'
    merged_df = pd.merge(df1, df2, on='en', how='outer', suffixes=('_df1', '_df2'))

    # Check the columns of the merged DataFrame
    print("Merged DataFrame columns:", merged_df.columns)


    # Initialize a dictionary to hold combined columns
    combined_data = {'en': merged_df['en']}

    # Combine language columns, keeping non-null values
    for col in lang_codes:
        df1_col = f'{col}_df1'
        df2_col = f'{col}_df2'
        if df1_col in merged_df.columns and df2_col in merged_df.columns:
            combined_data[col] = merged_df[[df1_col, df2_col]].bfill(axis=1).iloc[:, 0]
        elif df1_col in merged_df.columns:
            combined_data[col] = merged_df[df1_col]
        elif df2_col in merged_df.columns:
            combined_data[col] = merged_df[df2_col]

    # Get the list of probability columns that contain '_prob'
    prob_columns = [col.split('_df1')[0] for col in merged_df.columns if '_df1' in col and '_prob' in col]

    # Combine probabilities, taking the maximum where both are present
    for col in prob_columns:
        df1_col = f'{col}_df1'
        df2_col = f'{col}_df2'
        combined_data[col] = merged_df[[df1_col, df2_col]].max(axis=1)

    # Create a new DataFrame with combined data
    combined_df = pd.DataFrame(combined_data)
    return combined_df
# %%

save_dir = './data/TEST_SCRIPT2/'
os.makedirs(save_dir, exist_ok=True)
combined_bank = {}
for key, value in tqdm(bank.items()):
    combined_df = value[0]
    for df in value[1:]:
        combined_df = combine_dataframes(combined_df, df)
    
    # Remove rows with duplicate entries
    lang_present = [col for col in df.columns if col in lang_codes]
    combined_df.drop_duplicates(subset=lang_present, inplace=True)
    
    combined_bank[key] = combined_df
    combined_df.to_csv(os.path.join(save_dir, f'llama2_{key}.csv'), mode = "w", index=False)
# %%
