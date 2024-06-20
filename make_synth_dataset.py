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
import gen_data
from utils import plot_ci_plus_heatmap
from tuned_lens_wrap import load_tuned_lens
from dq_utils import proj, entropy, plot_ci, is_chinese_char, broadcast_kv_cache, printd
from logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap
from config_argparse import try_parse_args
import prefix
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
 # %%


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device,
                                                            dtype=cfg.model_dtype)
vocab = model.tokenizer.get_vocab()
# %%

    #{'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},

all_translation_banks = {'gemma' :
                            {'water': {'zh': '水', 'en': 'water', 'fr': 'eau', 'de': 'Wasser', 'ru': 'вода'},
                            'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'муж'},
                            'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'три'},
                            'new': {'zh': '新', 'en': 'village', 'fr': 'nouveau', 'de': 'neu', 'ru': 'пя'}},
                        'llama':
                            {'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},
                            'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'муж'},
                            'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'три'},
                            'new': {'zh': '新', 'en': 'village', 'fr': 'nouveau', 'de': 'neu', 'ru': 'пя'}}
                        }

# just use the same bank for both
translation_bank = all_translation_banks['llama']               
    
lang2name = {
    'fr': 'Français', 
    'de': 'Deutsch', 
    'en': 'English', 
    'zh': '中文', 
    'ru': 'Русский'
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
    


def check_bank_valid(bank):
    """
    Check if all tokens in the translation bank are in the vocabulary
    """    
    toks = sum([list(x.values()) for x in bank.values()],[])
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
              model = model, 
              translation_bank = translation_bank, 
              batch_size = None, 
              debug = False, 
              **kwargs):
    vocab = model.tokenizer.get_vocab()

    global __DEBUG__
    __DEBUG__ = debug

    if batch_size is None:
        batch_size = len(src_words)
    bs = batch_size
    
    threshold = 0

    
    # def run(src_words : List[str] , 
    #         src_lang : str, 
    #         dest_lang : str
    # ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        
    # TODO: Consider factoring out this code into a function
    # ====================================================
    all_probs = []
    all_toks = []
    
    prefix = gen_data.generate_translation_prompt(None, src_lang, dest_lang, translations = translation_bank)
    prefix_tok = model.tokenizer.encode(prefix, return_tensors="pt").to(device)
    
    kv_cache = prefix.gen_kv_cache(prefix_tok, model)
    
    suffixes = gen_data.generate_common_suffixes(src_words, src_lang, dest_lang) #suffixes will have leading space for gemma
    suffix_toks = prefix.tokenize_suffixes(suffixes, model.tokenizer)
    suffix_toks_batched = torch.split(suffix_toks, bs, dim=0)
    
    runner = tqdm(suffix_toks_batched, total=len(suffix_toks), desc=f"{src_lang} -> {dest_lang}", position=0, leave=True)
    
    for batch in runner:
        #broadcast_kv_cache(kv_cache, len(batch))
        #batch = batch.to(device)
        logits = prefix.run_with_kv_cache(batch, kv_cache, model)[:, -1].detach() # model returns (batch, seq, dvocab)
        probs = torch.softmax(logits, dim=-1)
        max_probs, max_tokens = torch.max(probs, dim=-1)
        
        all_probs.append(max_probs)
        all_toks.append(max_tokens)
        runner.update(len(batch))
        
    all_probs = torch.cat(all_probs, dim=0)
    all_toks = torch.cat(all_toks, dim=0)
    return all_probs, all_toks, suffix_toks, good_idx
    # ====================================================

    # if debug:
    #     for src_word in src_words:
    #         if is_chinese_char(src_word):
    #             assert src_word in vocab, f"Input zh string {src_word} not in vocabulary"
    #         else:
    #             assert "▁" + src_word in vocab, f"Input non-zh string {src_word} not in vocabulary"
    
    to_dest_probs, to_dest_tokens, src_suffix_toks, good_idx = run(src_words, src_lang, dest_lang)
    
    idx = (to_dest_probs > threshold) & good_idx.to(device)
    to_dest_probs = to_dest_probs[idx]
    to_dest_tokens = to_dest_tokens[idx]
    src_tokens = src_suffix_toks[idx,0]
    #print(f"{src_tokens.shape=}, {src_suffix_toks.shape=}")
    
    print(f"Kept {len(to_dest_probs)} / {len(idx)} translations")
    to_dest_words = model.tokenizer.convert_ids_to_tokens(to_dest_tokens)
    rev_src_probs, rev_src_tokens, dest_suffix_toks, good_idx2 = run(to_dest_words, dest_lang, src_lang)

    dest_tokens = dest_suffix_toks[:, 0]
    
    printd(model.tokenizer.convert_ids_to_tokens(src_tokens[:50]))
    printd(model.tokenizer.convert_ids_to_tokens(dest_tokens[:50]))
    printd(model.tokenizer.convert_ids_to_tokens(rev_src_tokens[:50]))

    to_dest_probs = to_dest_probs[good_idx2]
    to_dest_tokens = to_dest_tokens[good_idx2]
    src_tokens = src_tokens[good_idx2]

    cidx = (src_tokens == rev_src_tokens) & (rev_src_probs > threshold)
    
    print(f"{src_lang} = {dest_lang} Correct translations: {cidx.sum()} / {len(src_words)}")
    
    data = {
        src_lang: model.tokenizer.convert_ids_to_tokens(src_tokens[cidx]),
        dest_lang: model.tokenizer.convert_ids_to_tokens(dest_tokens[cidx]),
        src_lang + "_tok" : src_tokens[cidx].cpu(),
        dest_lang + "_tok" : dest_tokens[cidx].cpu(),
        f'{src_lang}_to_{dest_lang}_prob': to_dest_probs[cidx].cpu(),
        f'{dest_lang}_to_{src_lang}_prob': rev_src_probs[cidx].cpu()
    }

    df = pd.DataFrame(data)
    df = remove_dup_translation(df)
    return df

# df = translate(zh_tokens, "zh", "en", bs=128, debug=False, threshold =0)
# df
# %%
# %%
def en_tokens():
    en_words = []
    with open("./data/dict/en_dict.txt") as en_dict:
        for word in tqdm(en_dict):
            word = word.strip()
            if "▁" + word in vocab:
                en_words.append("▁" + word)
    print(f"Found {len(en_words)} in vocabulary from dictionary")
    return en_words
# %%


# %%
def verify_zh(df):
    df = df[df['zh'].apply(lambda x: all([is_chinese_char(c) for c in x]))]
    return df


def auto_translate(**kwargs):

    en_words = en_tokens()
    bank = {}

    print(f"Translation Bank: {translation_bank}")

    for lang in lang2name:
        if lang == 'en':
            continue
        bank[f"en_to_{lang}"] = translate(en_words, 'en', lang, **kwargs)
        
    bank["en_to_zh"] = verify_zh(bank["en_to_zh"]) # remove non-chinese characters

    df_all = gen_data.merge_dfs(bank.values())

    print(f"Merged size {len(df_all)}")
    print(df_all[:10])
    # Save filtered_df dataframe
    df_all = gen_data.remove_dups(df_all)
    bank["all"] = df_all
    return bank
# %%
# en_words = en_tokens()
# en_to_zh = translate(en_words, 'en', 'zh', batch_size=1)
#en_to_zh = translate(en_words[:200], 'en', 'zh', batch_size=64, debug=True)
# %%

def main(save_dir = None, **kwargs):
    assert save_dir is not None, "Please provide a save directory"
    os.makedirs(save_dir, exist_ok=True)
    
    bank = auto_translate(**kwargs)
    
    for key, df in tqdm(bank.items()):
        df.to_csv(os.path.join(save_dir, f'{key}.csv'), mode = "w", index=False)
# %%
cfg_dict = asdict(cfg)
main(**cfg_dict)
# %%