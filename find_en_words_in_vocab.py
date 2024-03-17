# %%
from dq_utils import tok_to_id
from transformers import AutoTokenizer
from tqdm import tqdm

tokenizer_llama = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf", 
                                                use_fast=False, add_prefix_space=False)


file_path = "data/dict/en_words_alpha.txt"
word_list = []

with open(file_path, "r") as file:
    for line in file:
        word = line.strip()
        word_list.append(word)
# %%

tokenizable_words = {}

for word in tqdm(word_list):
    (tok_id, new_word) = tok_to_id(word, tokenizer_llama)
    if tok_id is not None:
        tokenizable_words[word] = (tok_id, new_word)
        
# %% 