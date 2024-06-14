# %%
import warnings
import matplotlib.pyplot as plt
import torch
import pandas as pd
import os
from typing import List, Tuple

lang2name = {'en' : "English", 
            'zh' : "中文", 
            "fr" : "français", 
            "ru" : "pусский",
            "de" : "deutsch",
            "ko" : "한국어"}

def get_space_char(tokenizer):
    """
    Gets the space character used by the tokenizer.

    Args:
        tokenizer: The tokenizer object used for tokenization.

    Returns:
        str: The space character used by the tokenizer.
    """
    basename = tokenizer.__class__.__name__
    if basename == "QWenTokenizer":
        space_char = " "
    elif basename == "LlamaTokenizer":
        space_char = "▁"
    elif basename == "GPT2Tokenizer":
        space_char = "Ġ"
    else:
        raise NotImplementedError(f"get_space_char: Tokenizer {basename} not implemented")

    return space_char

def load_translations(base_path="data/langs"):
    translations = {}
    
    # List all language directories directly under base_path
    for lang_code in os.listdir(base_path):
        lang_path = os.path.join(base_path, lang_code, "clean.csv")
        
        # Check if the clean.csv file exists for the language
        if os.path.exists(lang_path):
            # Read the CSV file
            df = pd.read_csv(lang_path)
            
            # Loop through each row in the DataFrame
            for _, row in df.iterrows():
                word_eng = row['word_original']
                word_translated = row['word_translation']
                
                # If the English word is not in the dictionary, add it
                if word_eng not in translations:
                    translations[word_eng] = {}
                
                # Add the translation to the dictionary under the correct language
                translations[word_eng][lang_code] = word_translated

    return translations

def plot_logit_lens(logits_per_layer, tokenizer, ax, k=10):
    """
    Plots a heatmap of the top-k most likely tokens for each layer in a neural network.

    Args:
        logits_per_layer (torch.Tensor): Tensor of shape (num_layers, vocab_size) containing the logits for each layer.
        tokenizer (Tokenizer): Tokenizer object used to convert token IDs to token names.
        ax (matplotlib.axes.Axes): Axes object to plot the heatmap on.
        k (int, optional): Number of top tokens to consider. Defaults to 10.

    Returns:
        list: List of the most likely tokens for each layer.

    Raises:
        AssertionError: If the vocab size does not match the shape of the logits tensor.
    """
    n_layers, vocab_size = logits_per_layer.shape
    assert vocab_size == len(tokenizer.get_vocab()), "Mismatch between vocab size and logits shape"
    
    heatmap = torch.zeros(n_layers, k)
    token_names_matrix = []
    most_likely_tokens = []
    probs = torch.softmax(logits_per_layer, dim=-1).cpu()
    for j in range(n_layers):
        top_probs, top_tok = torch.topk(probs[j], k)
        heatmap[j] = top_probs
        token_names = tokenizer.convert_ids_to_tokens(top_tok)
        token_names_matrix.append(token_names)

        most_likely_tokens.append(token_names[0])

    # Plotting the heatmap
    plt.figure(figsize=(k, n_layers))
    ax = sns.heatmap(heatmap, annot=False, cmap='viridis', cbar=True)
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Layers')

    # Annotate each cell with the token name
    for i in range(n_layers):
        for j in range(k):
            plt.text(j + 0.5, i + 0.5, token_names_matrix[i][j].replace("Ġ","_"), fontsize=12,
                     ha='center', va='center', color='white', rotation=45)

    plt.show()
    return most_likely_tokens


# %%
# prompts = ['blue blue blue', 'the dog said']

# tokenizer.pad_token = tokenizer.eos_token
# logits_per_layer = logit_lens(model, prompts)
# fig, ax = plt.subplots(1, 1, figsize=(8, 16))
# plot_logit_lens(logits_per_layer[0], tokenizer, ax)


def filter_single_token_translations(translations, tokenizer, lang_codes):
    """
    Filters translations to include only those that are a single token according to the tokenizer and
    excludes translations identical to the English word or duplicates within the same set.
    
    :param translations: The dictionary of translations returned by the previous function.
    :param tokenizer: A tokenizer that supports convert_tokens_to_ids method.
    :param lang_codes: A list of language codes to filter the translations by.
    :return: A filtered dictionary of translations.
    """
    filtered_translations = {}
    space_char = get_space_char(tokenizer)
    
    for base_word, lang_translations in translations.items():
        filtered_lang_translations = {}
        seen_translations = set()  # Track seen translations to avoid duplicates
        
        for lang_code, translation in lang_translations.items():
            if lang_code in lang_codes and translation not in seen_translations and translation != base_word:
                # Convert translation to tokens and check if it's a single token
                token_ids = tokenizer.convert_tokens_to_ids(space_char + translation)
                if token_ids != tokenizer.unk_token_id:
                    filtered_lang_translations[lang_code] = translation
                    seen_translations.add(translation)
        
        if filtered_lang_translations:
            filtered_translations[base_word] = filtered_lang_translations
    
    return filtered_translations


def find_common_prefix(strs):
    """
    Finds the common prefix among a list of strings.

    Args:
        strs (list): A list of strings.

    Returns:
        str: The common prefix among the strings.

    """
    if not strs:
        return ""
    
    # The zip(*strs) transposes the list of strings, i.e., it groups the first character of each string together, then the second character, and so on.
    # This way, we can easily compare characters at the same position in all strings.
    for i, letter_group in enumerate(zip(*strs)):
        # If not all characters in this group are the same, we've found the end of the common prefix.
        if len(set(letter_group)) > 1:
            return strs[0][:i]
    # If we get through the entire loop, it means all characters in each position were the same for all strings, up to the length of the shortest string.
    return min(strs, key=len)



# %%



# def original_translation_dataset(tokenizer, langs=['fr'], base_path = "data/langs"):
#     space_char = get_space_char(tokenizer)

#     translation_dict = {lang: [] for lang in langs}
#     dfs = {lang: pd.read_csv(f"{base_path}/{lang}/clean.csv") for lang in langs}
    
    
#     # for i, row in df['en'].iterrows():
#     #     words = []
#     #     word_original = row['word_original']
#     #     words.append(word_original)
#     #     for lang in langs:
#     #         word = row[lang]
#     #         token_id = tokenizer.convert_tokens_to_ids(space_char + word)
#     #         if token_id == tokenizer.unk_token_id:
#     #             words = []
#     #             break
#     #         else:
#     #             words.append(word)
#     #             word_ids.append(token_id)
#     #     if words and len(set(words)) == len(words):
#     #         for lang, word in zip(langs, words):
#     #             translation_dict[lang].append(word)

#     print(f"Loaded {len(translation_dict[langs[0]])} out of {len(df)} examples")
#     return translation_dict

# %%
def gen_translation_dataset(path, tokenizer, langs=['fr', 'en']):
    """
    Generate a translation dataset from a given CSV file.

    Args:
        path (str): The path to the CSV file.
        tokenizer: The tokenizer object used for tokenization.
        langs (list): The list of languages to consider. Default is ['fr', 'en'].

    Returns:
        dict: A dictionary containing the translation dataset, where the keys are the languages
              and the values are lists of translated words.

    """
    space_char = get_space_char(tokenizer)

    translation_dict = {lang: [] for lang in langs}

    df = pd.read_csv(path)
    for i, row in df.iterrows():
        words = []
        word_ids = []
        for lang in langs:
            word = row[lang]
            token_id = tokenizer.convert_tokens_to_ids(space_char + word)
            if token_id == tokenizer.unk_token_id:
                words = []
                break
            else:
                words.append(word)
                word_ids.append(token_id)
        if words and len(set(words)) == len(words):
            for lang, word in zip(langs, words):
                translation_dict[lang].append(word)

    print(f"Loaded {len(translation_dict[langs[0]])} out of {len(df)} examples")
    return translation_dict


# dataset = gen_translation_dataset("data/test/single_tok_lang6.csv",
#                                   tokenizer, langs=['en', 'fr'])


# %%

def is_chinese_char(ch):
    """Check if a character is a Chinese character using a list of Unicode ranges and return range information.
    Now robust to invalid inputs."""
    try:
        c = ord(ch)
    except TypeError:
        return False
    
    # List of tuples, each representing a range of Chinese character code points with labels
    unicode_ranges = [
        (0x4E00, 0x9FFF, 'Common'),
        (0x3400, 0x4DBF, 'Extension A'),
        (0x20000, 0x2A6DF, 'Extension B'),
        (0x2A700, 0x2B73F, 'Extension C'),
        (0x2B740, 0x2B81F, 'Extension D'),
        (0x2B820, 0x2CEAF, 'Extension E'),
        (0x2CEB0, 0x2EBEF, 'Extension F'),
        (0x30000, 0x3134F, 'Extension G'),
        (0x31350, 0x323AF, 'Extension H'),
        (0xF900, 0xFAFF, 'CJK Compatibility Ideographs'),
        (0x2F800, 0x2FA1F, 'CJK Compatibility Ideographs Supplement')
    ]
    
    # Check if the character's code point falls within any of the ranges and return the range label
    for start, end, label in unicode_ranges:
        if start <= c <= end:
            return True
    return False


def plot_ci(data, ax, **kwargs):
    """
    Plots the mean and confidence interval of the given data on the specified axis.

    Parameters:
    - data: A tensor or array-like object containing the data.
    - ax: The axis object on which to plot the data.
    - **kwargs: Additional keyword arguments to be passed to the plot function.

    Returns:
    None
    """
    mean = data.mean(dim=1)
    std = data.std(dim=1)
    sem95 = 1.96 * std / (len(data)**0.5) 
    ax.plot(range(len(mean)), mean, **kwargs)
    ax.fill_between(range(len(mean)), mean - sem95, mean + sem95, alpha=0.3)
    

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


def raw_tok_to_id(string, tokenizer, add_space=False):
    """
    Converts a string to its corresponding token ID using the provided tokenizer.

    Args:
        string (str): The input string to be converted.
        tokenizer: The tokenizer object used for tokenization.
        add_space (bool, optional): Whether to add a space character before the string. Defaults to False.

    Returns:
        tuple: A tuple containing the token ID and the input string. If the token ID is None, the string could not be tokenized.

    Raises:
        NotImplementedError: If the tokenizer is not implemented.
    """
    assert type(string) == str, "Input must be a string"
    basename = tokenizer.__class__.__name__
    if tokenizer.is_fast:
        warnings.warn(f"Using {basename} with is_fast = True")

    if add_space:
        space_char = get_space_char(tokenizer)
        string = space_char + string

    if basename == "QWenTokenizer":
        string = string.encode('utf-8')
        try:
            token_id = tokenizer._convert_token_to_id(string)
        except ValueError:
            token_id = None

    elif basename == "LlamaTokenizer" or basename == "GPT2Tokenizer":
        token_id = tokenizer.convert_tokens_to_ids(string)
        if token_id == tokenizer.unk_token_id:
            token_id = None

    else:
        raise NotImplementedError(f"Tokenizer {basename} not implemented")

    return token_id, string


def tok_to_id(string, tokenizer):
    """
    Converts a string to its corresponding token ID using the provided tokenizer.
    If adding a space to the front of the string results in a valid token, the space is added.

    Args:
        string (str): The input string to be converted.
        tokenizer: The tokenizer object used for tokenization.
    Returns:
        tuple: A tuple containing the token ID and the modified string.
        If the token ID is None, the string could not be tokenized.
    """
    #     Note:
    #     Zero is not always the unknown token. It depends on the tokenizer implementation.
    #     For Llama-2, Mistral, 0 is <unk>.
    #     For Qwen, 0 is b"!".
    #     For gpt-2 and tinystories, 0 is "!". (unknown is 50256 for gpt-2/tinystories)
        
    space_char = get_space_char(tokenizer)
    
    token_id = raw_tok_to_id(string, tokenizer)
    if token_id is None:
        token_id = raw_tok_to_id(space_char + string, tokenizer)
        return token_id, space_char + string
    else:
        return token_id, string

def colour_encode(string, tokenizer):
    tokens = tokenizer.encode(string, add_special_tokens=False)
    print_tok(tokens, tokenizer)

def get_tok_prefix_ids(string, tokenizer, include_space = False, return_tensor = 'pt'):
    prefixes = [string[:i] for i in range(1, len(string) + 1)]
    if include_space:
        prefixes = [get_space_char(tokenizer) + prefix for prefix in prefixes] + prefixes
    valid_tokens = [x for x in prefixes if x in tokenizer.get_vocab()]
    prefix_ids = tokenizer.convert_tokens_to_ids(valid_tokens)
    if return_tensor == 'pt':
        return torch.tensor(prefix_ids)
    else:
        return prefix_ids
    

def print_tok(token_ids, tokenizer):
    """
    Print decoded tokens with rotating background colors.

    Parameters:
    - token_ids: List of token IDs to be decoded.
    - tokenizer: An object that has a .decode method for token IDs.
    """
    # List of ANSI escape codes for different background colors
    bg_colors = [
        "\033[41m",  # Red
        "\033[42m",  # Green
        "\033[43m",  # Yellow
        "\033[44m",  # Blue
        "\033[45m",  # Magenta
        "\033[46m",  # Cyan
    ]
    reset = "\033[0m"  # Reset styling

    # Cycle through the background colors as tokens are printed
    color_index = 0

    for token_id in token_ids:
        decoded_token = tokenizer.decode([token_id])
        print(f"{bg_colors[color_index]}{decoded_token}{reset}", end='')
        # Move to the next color, cycling back to the start if necessary
        color_index = (color_index + 1) % len(bg_colors)

# %%
# font = FontProperties(fname='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc')  # Specify the path to your font file
def plot_logit_lens(logits_per_layer, tokenizer, ax, k=10):
    """
    Plots a heatmap of the top-k most likely tokens for each layer in a neural network.

    Args:
        logits_per_layer (torch.Tensor): Tensor of shape (num_layers, vocab_size) containing the logits for each layer.
        tokenizer (Tokenizer): Tokenizer object used to convert token IDs to token names.
        ax (matplotlib.axes.Axes): Axes object to plot the heatmap on.
        k (int, optional): Number of top tokens to consider. Defaults to 10.

    Returns:
        list: List of the most likely tokens for each layer.

    Raises:
        AssertionError: If the vocab size does not match the shape of the logits tensor.
    """
    n_layers, vocab_size = logits_per_layer.shape
    assert vocab_size == len(tokenizer.get_vocab()), "Mismatch between vocab size and logits shape"
    
    heatmap = torch.zeros(n_layers, k)
    token_names_matrix = []
    most_likely_tokens = []
    
    for j in range(n_layers):
        probs = torch.softmax(logits_per_layer, dim=-1).cpu()
        top_probs, top_tok = torch.topk(probs, k)
        heatmap[j] = top_probs
        token_names = tokenizer.convert_ids_to_tokens(top_tok)
        token_names_matrix.append(token_names)

        most_likely_tokens.append(token_names[0])

    # Plotting the heatmap
    plt.figure(figsize=(k, n_layers))
    ax = sns.heatmap(heatmap, annot=False, cmap='viridis', cbar=True)
    ax.set_xlabel('Tokens')
    ax.set_ylabel('Layers')

    # Annotate each cell with the token name
    for i in range(n_layers):
        for j in range(k):
            plt.text(j + 0.5, i + 0.5, token_names_matrix[i][j], fontsize=12,
                     ha='center', va='center', color='white', rotation=45,fontproperties=font)

    plt.show()
    return most_likely_tokens