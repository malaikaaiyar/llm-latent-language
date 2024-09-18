# %%
from .constants import LANG2NAME
import torch
from typing import List
from collections import namedtuple

def gen_prompt(src_words = None, 
               dest_words = None, 
               src_lang = None, 
               dest_lang = None, 
               num_examples= None):
    """
    Generate a prompt for translation tasks.

    Args:
        src_words (list): List of source language words/phrases.
        dest_words (list): List of corresponding destination language words/phrases.
        src_lang (str): Source language code (e.g., 'fr' for French).
        dest_lang (str): Destination language code (e.g., 'zh' for Chinese).
        num_examples (int): Number of examples to include in the prompt (default: 1).

    Returns:
        str: The generated prompt string.
    """

    src_space = "" if src_lang == "zh" else " "
    dest_space = "" if dest_lang == "zh" else " "

    if num_examples is None:
        num_examples = len(dest_words)

    assert len(src_words) in [len(dest_words), len(dest_words)+1] , "Need N or N+1 source words for N dest words"

    prompt = ""
    for i in range(min(num_examples, len(src_words))):
        prompt += f'{LANG2NAME[src_lang]}: "{src_space}{src_words[i]}" - {LANG2NAME[dest_lang]}: "{dest_space}{dest_words[i]}"\n'

    # Add the last example without the destination translation
    prompt += f'{LANG2NAME[src_lang]}: "'

    return prompt
# %%

def gen_common_suffixes(src_words, 
                        src_lang = None, 
                        dest_lang = None):
    assert src_lang is not None, "Source language must be provided"
    assert dest_lang is not None, "Destination language must be provided"
    common_suffixes = []
    src_space = " " if src_lang != 'zh' else ""
    
    for src_word in src_words:
        src_word = src_word.split('▁')[-1] # Remove leading space token if present
        suffix = f'{src_space}{src_word}" {LANG2NAME[dest_lang]}: "'
        common_suffixes.append(suffix)
    return common_suffixes
        
TokenizedSuffixesResult = namedtuple('TokenizedSuffixesResult', 
                                     ['input_ids', 'attention_mask', 'indices'], 
                                     defaults=[None, None, None])
        

def tokenize_suffixes(suffixes : List[str], model):
    device = next(model.parameters()).device
    model.tokenizer.pad_token = model.tokenizer.eos_token
    suffix_tokens, attn_mask = model.tokenizer(suffixes,
                                                add_special_tokens=False,
                                                return_tensors="pt",
                                                padding=True).values()
    indices = attn_mask.sum(dim=1)-1
    assert torch.all(indices >= 0), "Attention mask has zeros, empty suffixes"
    suffix_tokens = suffix_tokens.to(device)
    
    return TokenizedSuffixesResult(
        input_ids=suffix_tokens,
        attention_mask=attn_mask,
        indices=indices
    )
    

# def generate_translation_prompt(word, src_lang=None, dest_lang=None, translations = translation_bank, **kwargs):
#     word = word.split('▁')[-1] if word is not None else None
    
#     src_space = " " if src_lang != 'zh' else ""
#     dest_space = " " if dest_lang != 'zh' else ""
#     not_dest_space = " " if dest_lang == 'zh' else ""

#     prompt = ""
#     for key, translation in translations.items():
#         prompt += f'{lang2name[src_lang]}: "{src_space}{translation[src_lang]}" {lang2name[dest_lang]}: "{dest_space}{translation[dest_lang]}"\n'
    
#     if word is None: #only generate common prefix
#         prompt += f'{lang2name[src_lang]}: "{src_space}'
#     else:
#         prompt += f'{lang2name[src_lang]}: "{src_space}{word}" {lang2name[dest_lang]}: "{not_dest_space}'
    
#     # Ensure prompt ends with a space for Chinese
#     # actually, no, we don't want this. It messes up the tokenization
#     # non-zh languages include the space. zh doesn't need the space.
#     # if dest_lang == 'zh':
#     #     prompt += ' '
    
#     return prompt