# %%
from transformers import GPT2Tokenizer
import os

# Load the GPT-2 tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

# Define the paths to the dictionary files
english_dict_path = 'data/dict/english.txt'
french_dict_path = 'data/dict/fr_dict.txt'
german_dict_path = 'data/dict/german.txt'

# Function to load words from a file into a set
def load_words_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        words = set(file.read().splitlines())
    return words

# Load words from the dictionary files into sets
english_words = load_words_from_file(english_dict_path)
french_words = load_words_from_file(french_dict_path)
german_words = load_words_from_file(german_dict_path)

# Initialize lists to store the filtered words
english_words_gpt2 = []
french_words_gpt2 = []
german_words_gpt2 = []

# Iterate over the GPT-2 vocabulary tokens
for token in tqdm(tokenizer.get_vocab().keys()):
    if not token.startswith("Ġ") or len(token) <= 3:
        continue
    word = token.lstrip("Ġ")  # Remove the leading "Ġ" symbol
    if word in english_words:
        english_words_gpt2.append(word)
    elif word in french_words and word not in english_words:
        print(token)
        french_words_gpt2.append(word)
    elif word in german_words and word not in english_words:
        german_words_gpt2.append(word)

# Create a dataset of English, French, and German words in the GPT-2 vocabulary
dataset = {
    'english': english_words_gpt2,
    'french': french_words_gpt2,
    'german': german_words_gpt2
}

# Print the number of words in each language
print(f"Number of English words in GPT-2 vocabulary: {len(english_words_gpt2)}")
print(f"Number of French words in GPT-2 vocabulary: {len(french_words_gpt2)}")
print(f"Number of German words in GPT-2 vocabulary: {len(german_words_gpt2)}")
# %%
