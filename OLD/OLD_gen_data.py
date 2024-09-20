# Generate clean data set of csv file of english, chinese, korean words, Each should be a single token

import os
import pandas as pd
from dataclasses import dataclass, field
import torch
from transformers import AutoTokenizer
from utils.misc import get_space_char, raw_tok_to_id, lang2name
import json

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

# read the csv files from each location
# %%

def gen_worddict(cfg):
    dfs = {}
    for lang in lang2name.keys():
        df_path = os.path.join(cfg.base_path, lang, "clean.csv")
        df = pd.read_csv(df_path)
        dfs[lang] = df
        
    word_dict = {}
    for lang, df in dfs.items():
        for index, row in df.iterrows():
            word = row['word_original']
            translation = row['word_translation']
            if word not in word_dict:
                word_dict[word] = {}
            word_dict[word][lang] = translation
    
    return word_dict

word_dict = gen_worddict(cfg)
list(word_dict.items())[0]
# %%
def filter_worddict(word_dict, cfg, tokenizer=tokenizer):
    """
    Filters the word dictionary for words that have a single token in all languages

    Args:
        word_dict (dict): The word dictionary containing translations for different languages.
        cfg (object): The configuration object containing language settings.
        tokenizer (object, optional): The tokenizer object used for tokenization. Defaults to tokenizer.

    Returns:
        dict: The filtered word dictionary.
    """
    
    new_word_dict = {}
    space_char = get_space_char(tokenizer)
    languages = [cfg.source_lang, cfg.target_lang, cfg.think_lang]
    for en_word in word_dict.keys(): #baseword always in english
        keepword = True
        
        #check translations exist for latent and target languages
        translated = [word_dict[en_word].get(lang) for lang in languages]
        if None in translated:
            print(f"Missing translation for {en_word}")
            continue
        
        #both target language and latent language should be a single token
        new_dict = {}
        for word, lang in zip(translated, languages):
            
            if lang == 'fr' or lang == 'en':
                word = space_char + word
                
            if lang == 'ko':
                #take first character
                word = word[0]
                
            tok_id, toked_word = raw_tok_to_id(word, tokenizer)
                
            if tok_id is not None:
                # print(f"Found {lang} token for {toked_word}, {tok_id}")
                new_dict[lang] = (toked_word, tok_id)
            else:
                print(f"{word} is not tokenizable")
                keepword = False
                break

        if keepword:
            print(f" {en_word} :  {new_dict}")
            new_word_dict[en_word] = new_dict
            keepword = False
        # for word, translations in new_word_dict.items():
        #     print(f"Kept word: {word}")
        #     for lang, (translation, token) in translations.items():
        #         print(f"Language: {lang}, Translation {translation} Token: {token}")
    return new_word_dict

filtered_word_dict = filter_worddict(word_dict, cfg, tokenizer)
print(f"Found {len(filtered_word_dict)}/{len(word_dict)} words single token translations.")

# %%
# Read the words from the file
file_path = 'data/test/single_tok_lang3.txt'
with open(file_path, 'r') as file:
    lines = file.readlines()

# Add the words to word_dict
for line in lines:
    words = line.strip().split(',')
    if len(words) == 3:
        en_word, zh_word, ko_word = words
        if en_word not in word_dict:
            word_dict[en_word] = {}
        word_dict[en_word][cfg.source_lang] = zh_word
        word_dict[en_word][cfg.target_lang] = ko_word
        word_dict[en_word][cfg.think_lang] = en_word

print(f"Added {len(lines)} words to word_dict.")

# Write word_dict to file
output_file = './data/word_dict.json'
with open(output_file, 'w') as file:
    json.dump(word_dict, file)

# Read the file back
with open(output_file, 'r') as file:
    loaded_word_dict = json.load(file)

# Check if word_dict and loaded_word_dict match
assert list(word_dict) == list(loaded_word_dict):
# %%
