import sys
import os
import numpy as np
import torch
from dataclasses import dataclass
import pandas as pd
from transformers import AutoTokenizer
from tqdm import tqdm
# %%

# Get the current working directory

# os.chdir("/root/llm-latent-language")
# print(f"Current Working Directory: {os.getcwd()}")

# %%

def merge_datasets(df_src, df_dest, vocab, cfg):
    """
    Process the dataset by filtering out rows that contain single tokens not present in the tokenizer's vocabulary.
    Then, merge the filtered source and destination dataframes based on the original word.

    Args:
        df_src (pandas.DataFrame): The source dataframe.
        df_dest (pandas.DataFrame): The destination dataframe.
        tokenizer (Tokenizer, optional): The tokenizer object used for tokenization. Defaults to tokenizer.
        cfg (Config, optional): The configuration object. Defaults to cfg.

    Returns:
        pandas.DataFrame: The merged dataframe containing the filtered data.
    """

    # this is expensive, only do it once and use the same vocab
    # DO NOT USE if x in tokenizer.get_vocab()

    src_lang = cfg.src_lang
    dest_lang = cfg.dest_lang
    debug = cfg.debug

    def keep_single_toks(df):
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
        debug and print(f'{count}/{len(df)} are single tokens')
        return new_df

        
    df_src = keep_single_toks(df_src)
    df_dest = keep_single_toks(df_dest)
    
    df_all = df_dest.merge(df_src, on=['word_original'], suffixes=(f'_{dest_lang}', f'_{src_lang}'))
    df_all.rename(columns={'word_original': 'en', 
                            f'word_translation_{dest_lang}': dest_lang, 
                            f'word_translation_{src_lang}': src_lang}, 
                            inplace=True)
    debug and print(f"Merged tokens: {len(df_all)}")
    return df_all
        

def find_all_tokens(token_str: str, vocab, cfg, return_tensors = "str"):
    """
    Given a string, find all tokens in the vocab that are prefixes of the string (with/without space)

    Args:
        token_str (str): The token string to search for tokens in.
        vocab (list): The vocabulary containing the valid tokens.

    Returns:
        list: A list of tokens found in the token string that exist in the vocabulary.
    """
    
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
            if leading_byte in vocab:
                return leading_byte
        return None
    
    def token_prefixes(token_str: str):
        return [token_str[:i] for i in range(1, len(token_str))]
    
    def add_spaces(tokens):
        return ['▁' + t for t in tokens]
    
    token_strs = [token_str]
    if cfg.token_add_prefixes:
        token_strs = token_strs +  token_prefixes(token_str)
    
    if cfg.token_add_spaces:
        token_strs = list(set(token_strs + add_spaces(token_strs)))
        
    final_tokens = [tok for tok in token_strs if tok in vocab]
    
    # just add leading byte for all languages unless it's in ascii range
    if cfg.token_add_leading_byte:
        tokid = unicode_leading_byte(token_str)
        if tokid is not None and tokid not in final_tokens:
            final_tokens.append(tokid)
    
    if return_tensors == "pt":
        return torch.LongTensor([vocab[x] for x in final_tokens])
    else:
        return final_tokens
    
        
    
# %%
# id2voc = {id:voc for voc, id in tokenizer.get_vocab().items()}
# def get_tokens(token_ids, id2voc=id2voc):
#     return [id2voc[tokid] for tokid in token_ids]

# def compute_entropy(probas):
#     probas = probas[probas>0]
#     return (-probas*torch.log2(probas)).sum(dim=-1)

lang2name = {'fr': 'Français', 'de': 'Deutsch', 'ru': 'Русский', 'en': 'English', 'zh': '中文'}
def gen_translation_task(df, vocab, cfg, return_tensors = "str"):
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
    
    src_lang = cfg.src_lang
    dest_lang = cfg.dest_lang
    latent_lang = cfg.latent_lang
    k = cfg.num_multi_shot
    dataset = []
    alt_dataset = []
    for ind in tqdm(range(len(df))):
        df = df.reset_index(drop=True)
        temp = df[df.index!=ind]
        sample = pd.concat([temp.sample(k), df[df.index==ind]], axis=0)
        prompt = ""
        for idx, (df_idx, row) in enumerate(sample.iterrows()):
            if idx < k-1:
                prompt += f'{lang2name[src_lang]}: "{row[src_lang]}" - {lang2name[dest_lang]}: "{row[dest_lang]}"\n'
            elif idx == k-1:
                prompt += f'{lang2name[src_lang]}: "{row[src_lang]}" - {lang2name[dest_lang]}: "'
                in_str, out_str, latent_str = row[src_lang], row[dest_lang], row[latent_lang]
                out_ids = find_all_tokens(out_str, vocab, cfg, return_tensors=return_tensors)
                latent_ids = find_all_tokens(latent_str, vocab, cfg, return_tensors=return_tensors)
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
                alt_latent_ids = find_all_tokens(alt_latent_str, vocab, cfg, return_tensors=return_tensors)
                alt_out_ids = find_all_tokens(alt_out_str, vocab, cfg, return_tensors=return_tensors)

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


def filter_correct(dataset, model, cfg):
    """
    Purges the dataset by removing instances that the mode doesn't predict correctly,
    both for the original and the counterfactual prompts.

    Args:
        dataset (list): The input dataset to be purged.
        model: The language model used for prediction.
        tokenizer: The tokenizer used to encode the input prompts.

    Returns:
        list: The purged dataset.
    """
    new_dataset = []
    device = next(model.parameters()).device
    correct = 0
    tokenizer = model.tokenizer
    runner = tqdm(dataset)
    for (i, d) in enumerate(runner):
        prompt = d['prompt']
        prompt_tok = tokenizer.encode(prompt, return_tensors='pt').to(device)
        y_guess = model(prompt_tok)[0, -1].argmax(-1).item()
        if y_guess not in d['out_ids']:
            continue
        cf_prompt = replace_source_word(prompt, d['alt_in_str'])
        cf_prompt_tok = tokenizer.encode(cf_prompt, return_tensors='pt').to(device)
        y_guess_cf = model(cf_prompt_tok)[0, -1].argmax(-1).item()
        if y_guess_cf in d['alt_out_ids']:
            correct += 1
            new_dataset.append(d)
        runner.set_description(f"filter_correct keeping: {correct}/{len(dataset)}")
    print(f"Filter dataset: {correct}/{len(dataset)} correct")
    return new_dataset
    
    
    