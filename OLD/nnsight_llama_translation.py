import os
import pandas as pd
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer
from dq_utils import get_space_char, get_tok_prefix_ids, raw_tok_to_id, lang2name, is_chinese_char, print_tok
from transformer_lens import HookedTransformer
import json
from tqdm import tqdm
import pandas as pd

torch.set_grad_enabled(False)

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
LOAD_MODEL = True
if 'model' not in locals() and LOAD_MODEL:
    model = HookedTransformer.from_pretrained(cfg.model_name, device=device)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)


new_df = pd.read_csv('data/test/single_tok_lang6.csv')

for index, row in new_df.iterrows():
    zh, en = row['zh'], row['en']
    assert zh in tokenizer.get_vocab(), f"Chinese word {zh} not in tokenizer vocab"
    
# %%
def colour_encode(string, tokenizer=tokenizer):
    tokens = tokenizer.encode(string, add_special_tokens=False)
    print_tok(tokens, tokenizer)


# %%

baseprompt = '中文:花 English: flower 中文:山 English: mountain 中文:月 English: moon 中文:水 English: water'
# Convert Chinese to English using Llama-2 model
converted_pairs = []
for index, row in tqdm(new_df.iterrows()):
    chinese = row['zh']
    en_answer = row['en'].split(" ")[0]
    en_answer_ids = get_tok_prefix_ids(en_answer, tokenizer, include_space = True)
    suffix = f'中文:{chinese} English:'
    prompt = baseprompt + suffix
    output = model(prompt)
    #en_guess = tokenizer.convert_ids_to_tokens(output[0,-1].argmax().item())
    #prob = torch.softmax(output[0,-1], dim=-1).max().item()
    en_guess = tokenizer.decode(output[0,-1].argmax().item())
    prob = torch.softmax(output[0,-1], dim=-1)
    en_prob = prob.max().item()
    en_cum_prob = prob[en_answer_ids].sum().item()
    
    print(f"zh: {chinese} en_true: {en_answer} en_guess: {en_guess} prob: {prob:.3f} cumprob: {en_cum_prob:.3f}")
    converted_pairs.append((chinese, en_answer, en_guess, prob))

    # converted_df = pd.DataFrame(converted_pairs, columns=['chinese', 'english_truth', 'english_guess', 'probability'])
    # converted_df.to_csv('data/test/converted_results.csv', index=False)
# %%

# %%


wrong_guesses = [pair for pair in converted_pairs if pair[2][1:] not in pair[1]]
for pair in wrong_guesses:
    print(f"{pair[0]} {pair[1]} {pair[2][1:]}")

# %%
import matplotlib.pyplot as plt

# Sort the converted pairs by probability in ascending order
sorted_pairs = converted_pairs.copy()
sorted_pairs.sort(key=lambda x: x[3])

# Extract the relevant data for plotting
chinese_words = [pair[0] for pair in sorted_pairs]
probabilities = [pair[3] for pair in sorted_pairs]
correct_guesses = [pair[2][1:] in pair[1] for pair in sorted_pairs]
colors = ['green' if correct else 'red' for correct in correct_guesses]

# Plot the probabilities
plt.figure(figsize=(10, 6))
plt.barh(chinese_words, probabilities, color=colors)
plt.xlabel('Probability')
plt.ylabel('Chinese Word')
plt.title('Llama-2 Language Conversion Probabilities')
plt.show()
# %%

failed_pairs = [pair for pair in converted_pairs if pair[2][1:] not in pair[1]]
sorted_failed_pairs = sorted(failed_pairs, key=lambda x: x[3], reverse=True)


# %%
chinese_characters = ' '.join(set(''.join([pair[0] for pair in converted_pairs])))
print(chinese_characters)
# %%
