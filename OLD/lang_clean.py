# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator
import csv

device = "cuda" # the device to load the model onto

#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

# %%
data = []
with open('lang_data.csv', 'r') as file:
    reader = csv.reader(file)
    for row in reader:
        english_word = row[0]
        french_word = row[1]
        chinese_word = row[2]
        korean_word = row[3]
        
        chinese_tokens = tokenizer.tokenize(chinese_word)
        korean_tokens = tokenizer.tokenize(korean_word)
        
        if len(chinese_tokens) == 2 and len(korean_tokens) == 2:
            data.append((english_word, french_word, chinese_word, korean_word))
# %%
