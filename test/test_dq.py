# %%
import dq

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from transformer_lens import HookedTransformer, utils
#!pip install git+https://github.com/callummcdougall/eindex.git
from eindex import eindex
from dq_utils import get_space_char, gen_translation_dataset
# %%

model_name = "gpt2"
model = HookedTransformer.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
# %%
langs = ["fr", "en"]

dataset = dq.gen_translation_dataset("data/test/single_tok_lang6.csv", tokenizer, langs)

prompt_dataset = {lang : dataset[lang][4:] for lang in langs}
test_dataset = {lang : dataset[lang][:4] for lang in langs}
# %%
# %%
import os
import pandas as pd

# %%
