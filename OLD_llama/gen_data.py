import sys
import os
import numpy as np
import torch
from dataclasses import dataclass
import pandas as pd
from transformers import AutoTokenizer
from tqdm.auto import tqdm
from typing import List, Dict, Any
from collections import namedtuple
# %%

# Get the current working directory

# os.chdir("/root/llm-latent-language")
# print(f"Current Working Directory: {os.getcwd()}")
#lang2name = {'fr': 'Français', 'de': 'Deutsch', 'ru': 'Русский', 'en': 'English', 'zh': '中文'}
lang2name = {'fr': 'Français', 'de': 'Deutsch', 'en': 'English', 'zh': '中文'}

all_translation_banks = {'gemma' :
                            {'water': {'zh': '水', 'en': 'water', 'fr': 'eau', 'de': 'Wasser', 'ru': 'вода'},
                            'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'мужчина'},
                            'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'пять'},
                            'new': {'zh': '新', 'en': 'new', 'fr': 'nouveau', 'de': 'neu', 'ru': 'новый'}},
                        'llama':
                            {'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},
                            'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'мужчина'},
                            'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'пять'},
                            'new': {'zh': '新', 'en': 'new', 'fr': 'nouveau', 'de': 'neu', 'ru': 'новый'}}
                        }

translation_bank = all_translation_banks['llama']

# def remove_prompt_overlap(df, prompt_bank = 0, src_lang = None, **kwargs):
#     src_words = translation_bank_extract(src_lang, prompt_bank=prompt_bank)
#     if src_lang != 'zh':
#         src_words = [f'▁{word}' for word in src_words]
#     df = df[~df[src_lang].isin(src_words)]
#     df = df.reset_index(drop=True)
#     return df 



# not as dumb as it looks
# 1/e chance of getting a derangement
# so only have to try a few times to get a derangement
def get_derangement(n):
    def is_derangement(x):
        return not (x == torch.arange(len(x))).any()
    while True:
        derangement = torch.randperm(n)
        if is_derangement(derangement):
            return derangement

# %%


def keep_single_toks(df, vocab):
        """
        Filter out rows in the DataFrame that contain words that are not present in the given vocabulary.

        Args:
            df (pandas.DataFrame): The input DataFrame containing the word translations.

        Returns:
            pandas.DataFrame: A new DataFrame with rows filtered based on the vocabulary.

        """
        count = 0
        new_df = df.copy()
        for idx, word in enumerate(new_df['word_translation']):
            if word in vocab or '▁'+word in vocab:
                count += 1
            else:
                new_df.drop(idx, inplace=True)
        print(f'{count}/{len(df)} are single tokens')
        return new_df

def all_dataset(vocab, **kwargs):
    df_langs = {}
    dataset_path = kwargs.get('dataset_path', './data/langs/')
    for lang in lang2name.keys():
        df_langs[lang] = pd.read_csv(os.path.join(dataset_path, lang, 'clean.csv'), usecols=['word_original', 'word_translation'])
        print(f"analysis {lang}")
        df_langs[lang] = keep_single_toks(df_langs[lang], vocab)
    merged_df = df_langs['en'].rename(columns={'word_translation': 'en'})
    # Merge each of the other DataFrames
    for lang, df in df_langs.items():
        if 'word_original' not in df.columns:
            print(f"Missing 'word_original' in the dataframe for {lang}")
        if lang != 'en':  # Skip the already initialized language
            # Perform the merge
            merged_df = merged_df.merge(
                df.rename(columns={'word_translation': lang}),
                on='word_original',
                how='inner'  # You can use 'inner' if you only want rows that exist in all languages
            )
    print(len(merged_df))
    return merged_df    


        
# %%
def remove_dups(df):
    """
    Remove duplicate rows from the given DataFrame.

    Args:
        df (pandas.DataFrame): The input DataFrame to remove duplicates from.

    Returns:
        pandas.DataFrame: The DataFrame with duplicates removed.
    """
    cols = [col for col in df.columns if col in lang2name]
    
    dup_row = []
    
    for idx, row in df.iterrows():
        if len(set([row[col].split('▁')[-1].lower() for col in cols])) < len(cols):
            dup_row.append(idx)
    df = df.drop(dup_row)
    return df.reset_index()

def merge_dfs(list_df, on = ['en', 'en_tok']):
    merged_df = list_df[0]
    for df in list_df[1:]:
        merged_df = merged_df.merge(df, on=on, how='inner')
    return merged_df

def load_dataset(dataset_path, src_lang, dest_lang, latent_lang):
    """
    Load and merge datasets for the given source, destination, and latent languages.

    Args:
        dataset_path (str): The path to the dataset directory.
        src_lang (str): The source language.
        dest_lang (str): The destination language.
        latent_lang (str): The latent language.

    Returns:
        pandas.DataFrame: The merged dataset with duplicates removed.
    """
    assert src_lang != dest_lang, "Source and destination languages must be different"
    datasets = {}
    for lang in [src_lang, dest_lang, latent_lang]:
        if lang in datasets or lang == 'en':
            continue
        lang_df = pd.read_csv(os.path.join(dataset_path, f'en_to_{lang}.csv'))
        datasets[lang] = lang_df
        
    datasets = list(datasets.values())
   
    all_df = merge_dfs(datasets)
    
    lang_col = [col for col in all_df.columns if col in lang2name]
    tok_col = [col for col in all_df.columns if '_tok' in col]
    other_col = [col for col in all_df.columns if col not in lang_col + tok_col]
    new_order = sorted(lang_col) + sorted(tok_col) + other_col 
    all_df = all_df[new_order]
    all_df = remove_dups(all_df)
    all_df = all_df.drop(columns="index")
    return all_df
    
TranslateCycleReturn = namedtuple('TranslateCycleReturn', 
                                  ['indices', 'src_tokens', 'dest_tokens', 'dest_probs', 'rev_probs'])
    
def translate_cycle(src_words, model, src_lang, dest_lang, **kwargs):
    device = next(model.parameters()).device
    trans_thresh = kwargs.get('trans_thresh', 0)
    keep_idx = torch.ones(len(src_words), dtype=torch.bool, device=device)
    to_dest_probs, dest_toks, src_toks, idx_to = prefix.run(src_words, model, src_lang, dest_lang, **kwargs)
    to_dest_strs = model.tokenizer.convert_ids_to_tokens(dest_toks)
    print('====')
    rev_src_probs, rev_src_tokens, df_dest_toks, idx_from = prefix.run(to_dest_strs, model, dest_lang, src_lang, **kwargs)
    
    # dataset has set of (src, dest, latent) triples
    dest_thresh_idx = (to_dest_probs > trans_thresh) # src -> dest translation probability is above threshold
    rev_thresh_idx = (rev_src_probs > trans_thresh) # dest -> src translation probability is above threshold
    rev_match_df = (src_toks == rev_src_tokens) # src -> dest' -> src' translation, src == src'
    #dest_match_df = (df_dest_toks == dest_toks) # src -> dest' -> src' translation, dest == dest'
    cidx = dest_thresh_idx & rev_thresh_idx & rev_match_df & idx_to & idx_from #& dest_match_df
    print(f"{src_lang} = {dest_lang} Correct translations: {cidx.sum()} / {len(cidx)}")

    # new_df = df[cidx.cpu().numpy()].copy()
    # new_df.loc[:, f'{src_lang}_to_{dest_lang}_prob'] = to_dest_probs[cidx].cpu().numpy()
    # new_df.loc[:, f'{dest_lang}_to_{src_lang}_prob'] = rev_src_probs[cidx].cpu().numpy()
    #new_df.reset_index(drop=True, inplace=True)
    return TranslateCycleReturn(cidx, src_toks, dest_toks, to_dest_probs, rev_src_probs)
    
def keep_correct(df, model, src_lang = None, dest_lang = None, trans_thresh = 0.5, **kwargs):
    device = next(model.parameters()).device

    cidx, _, dest_tok, dest_prob, rev_prob = translate_cycle(df[src_lang], model, src_lang, 
                                                          dest_lang, trans_thresh=trans_thresh, **kwargs)
    true_dest_tok = torch.LongTensor(df[f'{dest_lang}_tok'])
    cidx = cidx & (dest_tok == true_dest_tok) 
    print(f"{src_lang} = {dest_lang} Correct translations: {cidx.sum()} / {len(cidx)}")

    new_df = df[cidx.cpu().numpy()].copy()
    new_df.loc[:, f'{src_lang}_to_{dest_lang}_prob'] = dest_prob[cidx].cpu().numpy()
    new_df.loc[:, f'{dest_lang}_to_{src_lang}_prob'] = rev_prob[cidx].cpu().numpy()
    #new_df.reset_index(drop=True, inplace=True)
    return new_df

#keep_correct(raw_dataset, model, **cfg_dict)
# %%

# def merge_datasets(df_src, df_dest, df_latent, vocab, **kwargs):
#     """
#     Process the dataset by filtering out rows that contain single tokens not present in the tokenizer's vocabulary.
#     Then, merge the filtered source and destination dataframes based on the original word.

#     Args:
#         df_src (pandas.DataFrame): The source dataframe.
#         df_dest (pandas.DataFrame): The destination dataframe.
#         tokenizer (Tokenizer, optional): The tokenizer object used for tokenization. Defaults to tokenizer.
#         cfg (Config, optional): The configuration object. Defaults to cfg.

#     Returns:
#         pandas.DataFrame: The merged dataframe containing the filtered data.
#     """

#     # this is expensive, only do it once and use the same vocab
#     # DO NOT USE if x in tokenizer.get_vocab()

#     src_lang = kwargs.get('src_lang', 'fr')
#     dest_lang = kwargs.get('dest_lang', 'zh')
#     latent_lang = kwargs.get('latent_lang', 'en')
#     debug = kwargs.get('debug', False)
        
#     df_src = keep_single_toks(df_src, vocab)
#     df_dest = keep_single_toks(df_dest, vocab)
#     df_latent = keep_single_toks(df_latent, vocab)
    
#     df_merged = df_dest.merge(df_src, on=['word_original'], suffixes=('_dest', '_src'))
#     df_merged = df_merged.merge(df_latent, on=['word_original'])
#     df_merged.rename(columns={'word_original': 'en', 
#                               f'word_translation_dest': dest_lang, 
#                               f'word_translation_src': src_lang,
#                               'word_translation': latent_lang}, 
#                      inplace=True)
    
#     print(f"Merged tokens: {len(df_merged)}")
#     return df_merged

def unicode_leading_byte(token_str : str):
        """
        Returns the leading byte of a given token string if it is outside the ASCII range.

        Args:
            token_str (str): The token string to check.

        Returns:
            str or None: The leading byte of the token string if it is outside the ASCII range, None otherwise.
        """
        leading_byte = token_str.encode("utf-8")[0]
        if leading_byte >= 128: #outside ASCII range
            leading_byte = f'<0x{(token_str.encode("utf-8")[0]):X}>' # "好" -> "<0xE5>" 
            return leading_byte
        else:
            return None
    
def token_prefixes(token_str: str):
    return [token_str[:i] for i in range(1, len(token_str))]

def add_spaces(tokens):
    return ['▁' + t for t in tokens]        

def find_all_tokens(token_str: str, vocab, **kwargs):
    """
    Finds all valid tokens in a given token string based on the provided vocabulary.

    Args:
        token_str (str): The token string to search for tokens in.
        vocab (list): The vocabulary list containing valid tokens.
        **kwargs: Additional keyword arguments for customization.

    Keyword Args:
        token_add_prefixes (bool): Whether to add prefixes of the token string as tokens (default: True).
        token_add_spaces (bool): Whether to add tokens with spaces at the beginning (default: True).
        token_add_leading_byte (bool): Whether to add the leading byte of non-ASCII tokens as tokens (default: True).
        return_tensors (str): The type of tensors to return ('str' or 'pt', default: 'str').

    Returns:
        list or torch.Tensor: The list of valid tokens or a tensor of token indices.

    """
    if token_str[0] == '▁':
        token_str = token_str[1:]
    
    token_add_prefixes = kwargs.get('token_add_prefixes', True)
    token_add_spaces = kwargs.get('token_add_spaces', True)
    token_add_leading_byte = kwargs.get('token_add_leading_byte', True)
    return_tensors = kwargs.get('return_tensors', 'pt')
    
    token_strs = [token_str]
    if token_add_prefixes:
        token_strs = token_strs +  token_prefixes(token_str)
    
    if token_add_spaces:
        token_strs = list(set(token_strs + add_spaces(token_strs)))
        
    final_tokens = [tok for tok in token_strs if tok in vocab]
    
    # just add leading byte for all languages unless it's in ascii range
    if token_add_leading_byte:
        tokid = unicode_leading_byte(token_str)
        if tokid is not None and tokid not in final_tokens and tokid in vocab:
            final_tokens.append(tokid)
    
    if return_tensors == "str":
        return final_tokens
    else:
        return torch.LongTensor([vocab[x] for x in final_tokens])
        
    
        
    
# %%
# id2voc = {id:voc for voc, id in tokenizer.get_vocab().items()}
# def get_tokens(token_ids, id2voc=id2voc):
#     return [id2voc[tokid] for tokid in token_ids]

# def compute_entropy(probas):
#     probas = probas[probas>0]
#     return (-probas*torch.log2(probas)).sum(dim=-1)


# def filter_matching_translations(df):
#     # Identify columns that represent language translations by excluding probability columns
#     lang_cols = [col for col in df.columns if '_prob' not in col]
    
#     # Define a filter function to detect any rows with duplicate translations
#     def has_duplicate_translations(row):
#         # Check the row values for the language columns, if there are duplicates among them
#         translations = row[lang_cols].tolist()
#         return len(set(translations)) != len(translations)
    
#     # Apply the filter function to identify rows with duplicate translations
#     mask = df.apply(has_duplicate_translations, axis=1)
    
#     # Filter out the rows where any translations are duplicated
#     return df[~mask]

def gen_translation_task(df, vocab, **kwargs):
    """
    Generate a dataset for training a model using the given dataframe, vocabulary, and configuration.

    Args:
        df (pandas.DataFrame): The input dataframe containing the data.
        vocab (list): The vocabulary used for tokenization.
        cfg (Config): The configuration object containing the language settings and other parameters.

    Returns:
        list: A list of dictionaries, where each dictionary represents a datapoint in the dataset. Each dictionary contains the following keys:
            - 'prompt': The prompt string used for training.
            - 'out_ids': The token IDs of the output tokens.
            - 'out_str': The string representation of the output tokens.
            - 'latent_ids': The token IDs of the latent tokens.
            - 'latent_str': The string representation of the latent tokens.
            - 'in_str': The string representation of the input tokens.
    """
    src_lang = kwargs.get('src_lang', 'fr')
    dest_lang = kwargs.get('dest_lang', 'zh')
    latent_lang = kwargs.get('latent_lang', 'en')
    k = kwargs.get('num_multi_shot', 1)
    unique_prompt = kwargs.get('unique_prompt', True)

    seed = kwargs.get('seed', 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    dataset = []
    for ind in tqdm(range(len(df))):
        df = df.reset_index(drop=True)
        temp = df[df.index!=ind]
        sample = pd.concat([temp.sample(k), df[df.index==ind]], axis=0)
        prompt = ""
        src_space = "" if src_lang == "zh" else " "
        dest_space = "" if dest_lang == "zh" else " "
        for idx, (df_idx, row) in enumerate(sample.iterrows()):
            if idx < k-1:
                prompt += f'{lang2name[src_lang]}: "{src_space}{row[src_lang]}" - {lang2name[dest_lang]}: "{dest_space}{row[dest_lang]}"\n'
            elif idx == k-1:
                prompt += f'{lang2name[src_lang]}: "{src_space}{row[src_lang]}" - {lang2name[dest_lang]}: "'
                if dest_lang == 'zh':
                    prompt += ' '
                in_str, out_str, latent_str = row[src_lang], row[dest_lang], row[latent_lang]
                out_ids = find_all_tokens(out_str, vocab, **kwargs)
                latent_ids = find_all_tokens(latent_str, vocab, **kwargs)
                intersection = set(out_ids).intersection(set(latent_ids))
                if len(out_ids) == 0 or len(latent_ids) == 0:
                    print(f"Empty token ids for {in_str} -> {out_str}")
                    continue
                if dest_lang != 'en' and len(intersection) > 0:
                    print(f"Overlap in token ids for {in_str} -> {out_str}")
                    continue
            else:
                # Handling the steering additional row
                alt_in_str, alt_out_str, alt_latent_str = row[src_lang], row[dest_lang], row[latent_lang]
                alt_latent_ids = find_all_tokens(alt_latent_str, vocab, **kwargs)
                alt_out_ids = find_all_tokens(alt_out_str, vocab, **kwargs)

                datapoint = {'prompt': prompt, 
                    'idx' : ind,
                    'out_ids': out_ids, 
                    'out_str': out_str,
                    'latent_ids': latent_ids, 
                    'latent_str': latent_str, 
                    'in_str': in_str,
                    'alt_in_str': alt_in_str,
                    'alt_out_ids': alt_out_ids, 
                    'alt_out_str': alt_out_str,
                    'alt_latent_ids': alt_latent_ids, 
                    'alt_latent_str': alt_latent_str}
                dataset.append(datapoint)
    return dataset
# %%
def replace_source_word(prompt, new_french_word):
    # Split the prompt into lines
    lines = prompt.strip().split('\n')
    
    # Check if there are any lines to process
    if not lines:
        return prompt
    
    # Get the last line
    last_line = lines[-1]
    
    # Find the position of the last hyphen, which separates French and Chinese words
    hyphen_pos = last_line.rfind('-')
    
    # Replace the French word with the new word, keeping everything after the hyphen unchanged
    updated_last_line = f'Français: "{new_french_word}" ' + last_line[hyphen_pos:]
    
    # Replace the last line in the list with the updated line
    lines[-1] = updated_last_line
    
    # Join the lines back into a single string with new line characters
    updated_prompt = '\n'.join(lines)
    
    return updated_prompt


# def filter_correct(prompt, dataset, model, tokenizer):
#     """
#     Purges the dataset by removing instances that the mode doesn't predict correctly,
#     both for the original and the counterfactual prompts.

#     Args:
#         dataset (list): The input dataset to be purged.
#         model: The language model used for prediction.
#         tokenizer: The tokenizer used to encode the input prompts.

#     Returns:
#         list: The purged dataset.
#     """
#     prompt_tok = tokenizer.encode(prompt, return_tensors='pt')
#     generate_common_suffixes(dataset, tokenizer)
    
#     device = next(model.parameters()).device
#     correct = 0
#     tokenizer = model.tokenizer
#     runner = tqdm(dataset)
#     for (i, d) in enumerate(runner):
#         prompt = d['prompt']
#         prompt_tok = tokenizer.encode(prompt, return_tensors='pt').to(device)
#         y_guess = model(prompt_tok)[0, -1].argmax(-1).item()
#         if y_guess not in d['out_ids']:
#             #print("failed correct")
#             continue
#         cf_prompt = replace_source_word(prompt, d['alt_in_str'])
#         cf_prompt_tok = tokenizer.encode(cf_prompt, return_tensors='pt').to(device)
#         y_guess_cf = model(cf_prompt_tok)[0, -1].argmax(-1).item()
#         if y_guess_cf in d['alt_out_ids']:
#             correct += 1
#             new_dataset.append(d)
#         #else:
#         #    print("failed alt_correct")
#         runner.set_description(f"filter_correct keeping: {correct}/{len(dataset)}")
#     print(f"Filter dataset: {correct}/{len(dataset)} correct")
#     return new_dataset
    
    
    