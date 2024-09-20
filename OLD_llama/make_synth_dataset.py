# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm.auto import tqdm

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
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache

# %%
# ==== Custom Libraries ====
import OLD_llama.gen_data as gen_data
from utils.misc import is_chinese_char, printd
from utils.config_argparse import try_parse_args
import src.prefix as prefix
__DEBUG__ = True
# %%
torch.set_grad_enabled(False)
@dataclass
class Config:
    model_name: str = field(
        default='gemma-2b', 
        metadata={"help": "Model name to be used. Options include 'gemma-2b' and 'meta-llama/Llama-2-7b-hf'."})
    save_dir: str = field(
        default='DUMMY_NAME', 
        metadata={"help": "Directory to save data. Defaults to 'DUMMY_NAME' but should be set appropriately like './data/synth_gemma_2b' or './data/synth_llama_2_7b_new'."})
    model_dtype: str = field(default="fp16", 
                                     metadata={"help": "Data type of the model [fp16 | fp32 | auto]."})
    trans_thresh: float = field(default=0.0, 
                                metadata={"help": "Threshold for translations. Set to 0.0 by default."})
    batch_size: int = field(default=128, 
                            metadata={"help": "Batch size for processing. Default is 128."})


cfg = Config()
cfg = try_parse_args(cfg)
cfg.model_name = 'meta-llama/Llama-2-7b-hf'
 # %%


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device,
                                                            dtype=cfg.model_dtype)
vocab = model.tokenizer.get_vocab()
# %%

    #{'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},



# just use the same bank for both
translation_bank = gen_data.all_translation_banks['llama']               
    
lang2name = {
    'fr': 'Français', 
    'de': 'Deutsch', 
    'en': 'English', 
    'zh': '中文', 
}

# all_translation_bank = [
#     {'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},
#     'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'муж'},
#     'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'три'},
#     'new': {'zh': '新', 'en': 'village', 'fr': 'nouveau', 'de': 'neu', 'ru': 'пя'}},
    
#     {'water': {'zh': '水', 'en': 'water', 'fr': 'eau', 'de': 'Wasser', 'ru': 'вода'},
#     'middle': {'zh': '中', 'en': 'middle', 'fr': 'milieu', 'de': 'Mitte', 'ru': 'середина'},
#     'three': {'zh': '三', 'en': 'three', 'fr': 'trois', 'de': 'drei', 'ru': 'три'},
#     'woman': {'zh': '女', 'en': 'woman', 'fr': 'femme', 'de': 'Frau', 'ru': 'женщина'}}
# ]
    


def check_bank_valid(bank, valid_langs = lang2name.keys()):
    """
    Check if all tokens in the translation bank are in the vocabulary
    """   
    toks = sum([list(x[lang] for lang in valid_langs if lang in x) for x in bank.values()], [])
    new_toks = []
    for x in toks:
        if is_chinese_char(x):
            new_toks.append(x)
        else:
            new_toks.append("▁" + x)
                
    ids = [vocab.get(x,None) for x in new_toks]
    if None in ids:
        print(f"Invalid tokens: {ids}, {new_toks}")
        return False
    return True

assert check_bank_valid(translation_bank), f"Translation bank contains invalid tokens for model {model.cfg.model_name}"

# %%

def remove_dup_translation(df):
    # Filter columns list based on columns present in the DataFrame
    valid_columns = [col for col in lang2name if col in df.columns]
    
    # Apply a function across the DataFrame rows
    #filtered_df = df[df.apply(lambda row: len(set(row[valid_columns])) == len(valid_columns), axis=1)]
    filtered_df = df[df.apply(lambda row: len(set([str(x).lower() for x in row[valid_columns]])) == len(valid_columns), axis=1)]

    
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} duplicates for {valid_columns}")
    
    return filtered_df

    

# %%

@torch.no_grad()
def translate(src_words, 
              src_lang, 
              dest_lang, 
              model,
              debug = False, 
              **kwargs):
    vocab = model.tokenizer.get_vocab()

    global __DEBUG__
    __DEBUG__ = debug

    threshold = 0

    cidx, src_toks, dest_toks, dest_prob, rev_prob = gen_data.translate_cycle(src_words, 
                                                                              model, src_lang, dest_lang, **kwargs)

    print(f"{src_lang} = {dest_lang} Correct translations: {cidx.sum()} / {len(src_words)}")
    
    data = {
        src_lang: model.tokenizer.convert_ids_to_tokens(src_toks[cidx]),
        dest_lang: model.tokenizer.convert_ids_to_tokens(dest_toks[cidx]),
        src_lang + "_tok" : src_toks[cidx].cpu(),
        dest_lang + "_tok" : dest_toks[cidx].cpu(),
        f'{src_lang}_to_{dest_lang}_prob': dest_prob[cidx].cpu(),
        f'{dest_lang}_to_{src_lang}_prob': rev_prob[cidx].cpu()
    }

    df = pd.DataFrame(data)
    df = remove_dup_translation(df)
    # Throw out rows where the chinese translation isn't actually chinese characters
    if dest_lang == "zh":
        df = df[df['zh'].apply(lambda x: all([is_chinese_char(c) for c in x]))]

    return df

def en_tokens():
    en_words = []
    with open("./data/dict/en_dict.txt") as en_dict:
        for word in tqdm(en_dict):
            word = word.strip()
            if "▁" + word in vocab:
                en_words.append("▁" + word)
    print(f"Found {len(en_words)} in vocabulary from dictionary")
    return en_words

def auto_translate(**kwargs):

    en_words = en_tokens()
    bank = {}
    print(f"Translation Bank: {translation_bank}")

    for lang in lang2name:
        if lang == 'en':
            continue
        bank[f"en_to_{lang}"] = translate(en_words, 'en', lang, model, **kwargs)

    df_all = gen_data.merge_dfs(list(bank.values()))

    print(f"Merged size {len(df_all)}")
    print(df_all[:10])
    # Save filtered_df dataframe
    df_all = gen_data.remove_dups(df_all)
    bank["all"] = df_all
    return bank

# en_words = en_tokens()
# en_to_zh = translate(en_words, 'en', 'zh', batch_size=1)
#en_to_zh = translate(en_words[:200], 'en', 'zh', batch_size=64, debug=True)


def main(save_dir = None, **kwargs):
    assert save_dir is not None, "Please provide a save directory"
    os.makedirs(save_dir, exist_ok=True)
    
    bank = auto_translate(**kwargs)
    
    for key, df in tqdm(bank.items()):
        df.to_csv(os.path.join(save_dir, f'{key}.csv'), mode = "w", index=False) 
    return bank

cfg_dict = asdict(cfg)
bank = main(**cfg_dict)
# %%