# %%
%load_ext autoreload
%autoreload 2
# %%

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

os.environ["TRANSFORMERS_CACHE"] = "~/rds/rds-dsk-lab-eWkDxBhxBrQ/transformers_cache"
os.environ["HF_HOME"] = "~/rds/rds-dsk-lab-eWkDxBhxBrQ/transformers_cache"

# %%
LOAD_MODEL = True
if 'model' not in locals() and LOAD_MODEL:
    model = HookedTransformer.from_pretrained(cfg.model_name, device=device)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
# %%
# Read data/dict/francias.txt into a data structure that is fast to check membership of
fr_dict_raw = set()
with open('data/dict/fr_dict.txt', 'r') as file:
    for line in file:
        word = line.strip().lower()
        fr_dict_raw.add(word)

en_dict = set()
with open('data/dict/english.txt', 'r') as file:
    for line in file:
        word = line.strip().lower()
        en_dict.add(word)
        
fr_dict = fr_dict_raw - en_dict

#%%
fr_tokens = []
for raw_word in tqdm(tokenizer.get_vocab().keys()):
    leading_space = raw_word[0] == '▁'
    long_enough = len(raw_word) > 2
    in_dict = raw_word[1:] in fr_dict
    if leading_space and long_enough and in_dict:
        fr_tokens.append(raw_word)
print(f"Found {len(fr_tokens)} tokens in the French dictionary")
        
        
spell_prompt = ""
    
# %%
for x in fr_tokens:
    print(x[1:])
# %%
french_prompt_words = ["musée", "nombre", "novembre", "avec"]
english_prompt_words = ["museum", "number", "november", "with"]

def gen_spelling_prompt(words):
    base_prompt = ""
    for word in words:
        base_prompt += " ".join(word) + ": " + word + " "
    return base_prompt.strip()
base_prompt = gen_spelling_prompt(french_prompt_words)
# %%

fr_to_en_pairs = []
df = pd.read_csv('data/test/fr_to_en.csv')
for i, row in df.iterrows():
    fr = row['fr']
    en = row['en']
    fr_id = tokenizer.convert_tokens_to_ids("▁" + fr)
    en_id = tokenizer.convert_tokens_to_ids("▁" + en)
    if fr_id != tokenizer.unk_token_id and en_id != tokenizer.unk_token_id:
        fr_to_en_pairs.append((fr, fr_id, en, en_id))
print(f"Loaded {len(fr_to_en_pairs)} examples")
# %%

#  = []
# fr_answer = []
# en_answer = []

# for fr, fr_id, en, en_id in fr_to_en_pairs:
#     prompt = base_prompt + " " + " ".join(fr) + ":"
#     prompts.append(prompt)
#     fr_answer.append(fr_id)
#     en_answer.append(en_id)    
# %%
def gen_translation_prompt(src_words, dest_words, src_lang="français", dest_lang="English", new_word=None):
    """
    Generate a translation prompt based on source and destination words.

    Args:
        src_words (list): List of source words.
        dest_words (list): List of destination words.
        src_lang (str, optional): Source language. Defaults to "français".
        dest_lang (str, optional): Destination language. Defaults to "English".
        new_word (str, optional): Additional word to include in the prompt. Defaults to None.

    Returns:
        str: The generated translation prompt.
    """
    base_prompt = ""
    for src, dest in zip(src_words, dest_words):
        base_prompt += f"{src_lang}: {src} {dest_lang}: {dest} "
    if new_word is not None:
        base_prompt += f"{src_lang}: {new_word} {dest_lang}:"
    return base_prompt.strip()
# %%
translation_base_prompt = gen_translation_prompt(french_prompt_words, english_prompt_words)
translation_prompts = []
en_id_answers = []
for fr, fr_id, en, en_id in fr_to_en_pairs:
    prompt = translation_base_prompt + " français: " + fr + " English:"
    translation_prompts.append(prompt)
    en_id_answers.append(en_id)    

# %%
# from nnsight import LanguageModel
# import os
# nnsight_model = LanguageModel("meta-llama/Llama-2-7b-hf", 
#                               device_type = "cuda",
#                               use_auth_token = "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")
# print(nnsight_model)



"""
TODO FIX THIS SHIT
"""

def latent_lang(prompts, model, id_family):
    probs = [torch.zeros(len(prompts), model.config.n_layers) for _ in range(len(id_family))]
    for i, prompt in enumerate(prompts):
        output, cache = model.run_with_cache(prompt)
        for j in range(model.config.n_layers):
            
            
            resid = cache[f'blocks.{j}.hook_resid_post'] 
            ln_resid = model.ln_final(resid)
            logits = model.unembed(ln_resid)
            #logits = ln_resid[0, -1, :] @ model.unembed.W_U + model.unembed.b_U
            zh_prob = torch.softmax(logits[0, -1, :], dim=-1)[zh_answer_ids[i]].item()
            es_prob = torch.softmax(logits[0, -1, :], dim=-1)[es_answer_ids[i]].item()
        
        zh_probs[j, i] = zh_prob
        es_probs[j, i] = es_prob
            
            resid = cache[j]
            for k, ids in enumerate(id_family):
                probs[k][i, j] = torch.softmax(output[0, ids, j], dim=-1).max()

# %%
en_guesses = []
en_answers = []
en_probs = []
for i, prompt in tqdm(enumerate(translation_prompts)):
    #en_answer_ids = get_tok_prefix_ids(en_answer, tokenizer, include_space = True)
    output = model(prompt)
    #en_guess = tokenizer.convert_ids_to_tokens(output[0,-1].argmax().item())
    #prob = torch.softmax(output[0,-1], dim=-1).max().item()
    en_guess = tokenizer.decode(output[0,-1].argmax().item())
    prob = torch.softmax(output[0,-1], dim=-1)
    en_prob = prob.max().item()
    french = fr_to_en_pairs[i][0]
    en_answer = fr_to_en_pairs[i][2]
    #en_cum_prob = prob[en_answer_ids].sum().item()
    
    print(f"fr: {french} en_true: {en_answer} en_guess: {en_guess} prob: {en_prob:.3f}")
    en_guesses.append(en_guess)
    en_probs.append(en_prob)
    en_answers.append(en_answer)
    #converted_pairs.append((chinese, en_answer, en_guess, prob))

# %%
# failed_examples = [(fr, en_guess, en_answer) for fr, en_guess, en_answer in zip(fr_to_en_pairs, en_guesses, en_answers) if en_guess[1:] != en_answer]
# for fr, en_guess, en_answer in failed_examples:
#     print(f"| French    | English (True) | English (Guess) |")
#     print(f"|-----------|----------------|----------------|")
#     for fr, en_guess, en_answer in failed_examples:
#         print(f"| {fr[0]:<10} | {en_answer:<14} | {en_guess:<14} |")
# %%
french_words = [fr_to_en_pairs[i][0] for i in range(len(fr_to_en_pairs))]
spelling_prompts = [gen_spelling_prompt(french_prompt_words, english_prompt_words,
                                        new_word = fr) for fr in french_words]


translation_prompts = []
en_id_answers = []
for fr, fr_id, en, en_id in fr_to_en_pairs:
    prompt = translation_base_prompt + " français: " + fr + " English:"
    translation_prompts.append(prompt)
    en_id_answers.append(en_id)    