# %%
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass, field
torch.set_grad_enabled(False)
import warnings
import langdetect
import langid
import pandas as pd
from langdetect import detect
import dq
import eindex
from transformers import LlamaForCausalLM, LlamaTokenizer, LogitsProcessor
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import einops
import pickle
from eindex import eindex
import time
from llama_utils import measure_top_k_accuracy
# %%
# cfg = Config()
# cfg.model_kwargs = {'use_fast': False, 'add_prefix_space': False}
# # Set torch device to use CPU only
# device = torch.device('cpu')
# tokenizer = HookedTransformer.from_pretrained(cfg.model_name, device=device).tokenizer
torch.set_grad_enabled(False)
# # # Replace 'llama-2-model-name' with the actual model name for Llama-2
# # tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **cfg.model_kwargs)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model_name = "meta-llama/Llama-2-13b-hf"
# model = AutoModelForCausalLM.from_pretrained(model_name, load_as_8bit=True, low_cpu_mem_usage=True).to(device)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_name = "meta-llama/Llama-2-7b-hf"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_prefix_space=False)
model = HookedTransformer.from_pretrained_no_processing(model_name, dtype=torch.float16, device='cuda:0', low_cpu_mem_usage=True)



# %%

# Read the Chinese text from file
chinese_text = []
with open('data/story/llama_zh_2.txt', 'r', encoding='utf-8') as file:
    chinese_text = file.readlines()
chinese_text = [line.strip() for line in chinese_text]
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
chinese_tokens = tokenizer(chinese_text, return_tensors="pt", padding=True).input_ids.to(device)
# %%
# Tokenize the Chinese text

measure_top_k_accuracy(chinese_text, model, tokenizer, top_k_values=[1, 5, 10])

# %%

def plot_sorted_probs(predictions, test_to_predict):
    import matplotlib.pyplot as plt

    probs_per_index = eindex(predictions, test_to_predict, "b seq [b seq]").cpu()

    # Sort probs_per_index for each sequence
    sorted_probs_per_index = [sorted(seq_probs) for seq_probs in probs_per_index]

    # Plot sorted_probs_per_index for each sequence
    for i, seq_probs in enumerate(sorted_probs_per_index):
        plt.figure()
        plt.bar(range(len(seq_probs)), seq_probs)
        plt.xlabel('Index')
        plt.ylabel('Probability')
        plt.title(f'Probability per Index (Sorted) - Sequence {i+1}')
        plt.yscale('log')
        plt.show()

#plot_sorted_probs(predictions, test_to_predict)
# %%
# Load the dictionary from file
with open('data/llama/zh_en_vocab.pkl', 'rb') as f:
    llama_langs = pickle.load(f)
# %%


en_ids, zh_ids = llama_langs['en_ids'], llama_langs['zh_ids']

for tok_prompt in chinese_tokens:
    output, cache = model.run_with_cache(tok_prompt):
    hidden_layers=  []        



"""
TOOD FINISH THIS THING
"""
@torch.no_grad
def per_layer_prediction_loss(prompt, model, tokenizer):
    if not isinstance(prompt, torch.Tensor):
        prompt = tokenizer.encode(prompt, return_tensors="pt")
# %%
