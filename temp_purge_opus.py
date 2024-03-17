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
df = pd.read_csv('data/test/opus.csv')
df = df[df['chinese'].isin(tokenizer.get_vocab())]
df.to_csv('data/test/opus_llama.csv', index=False)
# %%


df_llama = pd.read_csv('data/test/converted_results.csv')
merged_df = pd.merge(df, df_llama, on='chinese', how='inner')
filtered_df = merged_df[~((merged_df['english'] == merged_df['english_truth']) & (merged_df['english_guess'].str[:1].isin(merged_df['english_truth'])))]
print(filtered_df)

# %%
duplicated_df = merged_df[merged_df['chinese'].duplicated(keep=False)]
print(duplicated_df)
# %%
