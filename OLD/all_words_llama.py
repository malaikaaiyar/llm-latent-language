import os
import pandas as pd
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer
from dq_utils import get_space_char, raw_tok_to_id, lang2name, is_chinese_char
from transformer_lens import HookedTransformer
import json
from tqdm import tqdm
import pandas as pd

# %%
# Reload libraries in Jupyter Notebook
%load_ext autoreload
%autoreload 2
# %%
@dataclass
class Config:
    source_lang: str = 'zh'
    target_lang: str = 'ko'
    think_lang: str = 'en'
    model_name: str = 'meta-llama/Llama-2-7b-hf'
    base_path: str = 'data/langs/'
    model_kwargs: dict = field(default_factory=dict)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cfg = Config()
cfg.model_kwargs = {'use_fast': False, 'add_prefix_space': False}

tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **cfg.model_kwargs) 

device = torch.device('cpu')

# %%

if 'model' not in locals():
    model = HookedTransformer.from_pretrained(cfg.model_name, device=device)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


prompt = '中文:花 english: flower 中文:山 english: mountain 中文:月 english: moon 中文:水 english: water'
# %%
vocab = tokenizer.get_vocab()
new_vocab = [x for x in vocab if is_chinese_char(x)[0]]
print(' '.join(new_vocab))
# %%
# all_tokens = tokenizer.get_vocab().keys()
# dictionary_path = './data/dict/en_words_alpha.txt'
# with open(dictionary_path, 'r') as f:
#     dictionary_words = set(f.read().splitlines())
    
# word_dict = {}
# for tok in tqdm(tokenizer.get_vocab()):
#     if tok[0] == get_space_char(tokenizer) and tok[1:] in dictionary_words:
#         word_dict[tok] = tokenizer.get_vocab()[tok]

# output_file = 'data/dict/llama-2-zh-tokens.txt'
# with open(output_file, 'w') as f:
#     for char in new_vocab:
#         f.write(char + '\n')
# %%
# %%
csv_file = 'data/test/zh_en_claude.csv'
df = pd.read_csv(csv_file)

same_english_pairs = {}
for index, row in df.iterrows():
    english = row['english']
    chinese = row['chinese']
    if english in same_english_pairs:
        same_english_pairs[english].append(chinese)
    else:
        same_english_pairs[english] = [chinese]

# Print the pairs with the same English word and their Chinese symbols
for english, chinese_symbols in same_english_pairs.items():
    if len(chinese_symbols) > 1:
        print(f"{english}", end=" ")
        for chinese in chinese_symbols:
            print(f"{chinese}", end=" ")
        print()

disambiguate_file = 'data/test/disambiguate.csv'
zh_en_claude_file = 'data/test/zh_en_claude.csv'

disambiguate_df = pd.read_csv(disambiguate_file)
zh_en_claude_df = pd.read_csv(zh_en_claude_file)


new_pairs = []
seen_english_words = set()
for index, row in zh_en_claude_df.iterrows():
    english = row['english']
    chinese = row['chinese']
    if english in disambiguate_df['english'].values:
        chinese = disambiguate_df.loc[disambiguate_df['english'] == english, 'chinese'].values[0]
    if english not in seen_english_words:
        new_pairs.append((english, chinese))
        seen_english_words.add(english)

new_df = pd.DataFrame(new_pairs, columns=['english', 'chinese'])
# %%
new_df.to_csv('data/test/zh_en_claude_disambiguated.csv', index=False)

# %%
