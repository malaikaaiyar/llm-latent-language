# %%
from transformers import AutoModelForCausalLM, AutoTokenizer
from googletrans import Translator

device = "cuda" # the device to load the model onto

#model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
from IPython.display import HTML, display
from tqdm import tqdm
# %%
    
# Read language data from single_tok_lang.txt
filtered_data = []
with open('single_tok_lang.txt', 'r') as file:
    for line in file:
        english, french, chinese, korean = line.strip().split(',')
        filtered_data.append((english, french, chinese, korean))
    
single_token_data = []
for english_word, french_symbol, chinese_symbol, korean_symbol in filtered_data:
    english_tokens = tokenizer.tokenize(english_word)
    chinese_tokens = tokenizer.tokenize(chinese_symbol)
    korean_tokens = tokenizer.tokenize(korean_symbol)
    # print tokens
    print(english_tokens, chinese_tokens, korean_tokens)
    if len(english_tokens) == 1 and len(chinese_tokens) == 2 and len(korean_tokens) == 2:
        single_token_data.append((english_word, french_symbol, chinese_symbol, korean_symbol))


# %%
# Read words from simple_english_words.txt and put in a list
word_list = []
tokenized_word_list = []
single_token_words = []

with open('simple_english_1000.txt', 'r') as file:
    for line in file:
        word = line.strip()
        word_list.append(word)
        
for word in tqdm(word_list):
    tokens = tokenizer.tokenize(word)
    tokenized_word_list.append(tokens)
    if len(tokens) == 1:
        single_token_words.append(word)
        
# %%



# %%
# Forward pass on the model and return the status of the residual stream after each hidden layer
example_prompt = prompts[0]
input_ids = tokenizer(example_prompt, return_tensors="pt", padding=True).input_ids.to(device)
attention_mask = tokenizer(example_prompt, return_tensors="pt", padding=True).attention_mask.to(device)

outputs = model(input_ids=input_ids, attention_mask=attention_mask)

residuals = outputs["residuals"]

# Print the status of the residual stream after each hidden layer
for i, residual in enumerate(residuals):
    print(f"Hidden Layer {i+1}: {residual}")

# %%
