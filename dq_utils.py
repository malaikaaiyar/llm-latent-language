
import warnings
import matplotlib.pyplot as plt
import torch

lang2name = {'en' : "English", 
            'zh' : "中文", 
            "fr" : "français", 
            "ru" : "pусский",
            "de" : "deutsch",
            "ko" : "한국어"}

def is_chinese_char(ch):
    """Check if a character is a Chinese character using a list of Unicode ranges and return range information.
    Now robust to invalid inputs."""
    try:
        c = ord(ch)
    except TypeError:
        return False, "Invalid input: Not a character"
    
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
            return True, label
    return False, "Not chinese"


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
        raise NotImplementedError(f"Tokenizer {basename} not implemented")

    return space_char


def raw_tok_to_id(string, tokenizer, add_space=False):
    """
    Converts a string to its corresponding token ID using the provided tokenizer.

    Args:
        string (str): The input string to be converted.
        tokenizer: The tokenizer object used for tokenization.
    Returns:
       The Token ID. If the token ID is None, the string could not be tokenized.

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

        token_id =  tokenizer.convert_tokens_to_ids(string)     
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
