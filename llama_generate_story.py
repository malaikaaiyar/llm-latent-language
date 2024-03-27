# %%
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
import pandas as pd
from dataclasses import dataclass, field
torch.set_grad_enabled(False)
import warnings
import langdetect
import langid
import pandas as pd
from langdetect import detect
import dq
import random
from transformers import LlamaForCausalLM, LlamaTokenizer, LogitsProcessor
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers import AutoTokenizer
# %%
# cfg = Config()
# cfg.model_kwargs = {'use_fast': False, 'add_prefix_space': False}
# # Set torch device to use CPU only
# device = torch.device('cpu')
# tokenizer = HookedTransformer.from_pretrained(cfg.model_name, device=device).tokenizer
torch.set_grad_enabled(False)
# # # Replace 'llama-2-model-name' with the actual model name for Llama-2
# # tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **cfg.model_kwargs)
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model_name = "meta-llama/Llama-2-13b-hf"
# model = AutoModelForCausalLM.from_pretrained(model_name, load_as_8bit=True, low_cpu_mem_usage=True).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, add_prefix_space=False)




device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# model_name = "meta-llama/Llama-2-13b-hf"

# Configure the model to use 8-bit inference
quantization_config = BitsAndBytesConfig(load_in_8bit=True, llm_int8_threshold=6.0)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda:0",
    quantization_config=quantization_config,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)


# %%

# input_text = "曾几何时，在一个很远很远的地方，"
# input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
# max_length = 50
# num_return_sequences = 1
# temperature = 0.8

# generated_text = model.generate(
#     input_ids=input_ids,
#     max_length=max_length,
#     num_return_sequences=num_return_sequences,
#     temperature=temperature,
# )

# # Decode the generated text
# generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True)
# print(generated_text)

# %%

# Load English dictionary into a set
english_dict = set()
with open('data/dict/english.txt', 'r') as file:
    for line in file:
        word = line.strip()
        english_dict.add(word)
# %%


chinese_tokens = []
chinese_tokens_ids = []
english_tokens = []
english_tokens_ids = []
for token, tok_id in tqdm(tokenizer.get_vocab().items()):
    if dq.is_chinese_char(token):
        chinese_tokens.append(token)
        chinese_tokens_ids.append(tok_id)
    basetoken = token.lstrip("▁")
    if basetoken in english_dict:
        english_tokens.append(token)
        english_tokens_ids.append(tok_id)
       
zh_punctuation = "，。！？、；：“”‘’（）"
for tok in zh_punctuation:
    if tok in tokenizer.get_vocab():
        chinese_tokens.append(tok)
        chinese_tokens_ids.append(tokenizer.get_vocab()[tok])
        
en_punctuation = ",.!?;:\"'()"

for tok in en_punctuation:
    if tok in tokenizer.get_vocab():
        english_tokens.append(tok)
        english_tokens_ids.append(tokenizer.get_vocab()[tok])

print(f"Number of Chinese tokens: {len(chinese_tokens)}/{len(tokenizer.get_vocab())}")
print(f"Number of English tokens: {len(english_tokens)}/{len(tokenizer.get_vocab())}")

# %%

zh_tok_ids = torch.tensor(chinese_tokens_ids)
en_tok_ids = torch.tensor(english_tokens_ids)
    
    
vocab_dict = {
    'zh_ids': chinese_tokens_ids,
    'en_ids': english_tokens_ids,
    'zh': chinese_tokens,
    'en': english_tokens
}

import pickle
# Save the dictionary to a file
with open('data/llama/zh_en_vocab.pkl', 'wb') as file:
    pickle.dump(vocab_dict, file)
    

# %%
from transformers import LogitsProcessor, RepetitionPenaltyLogitsProcessor
import os
import pickle


class RestrictedTokensLogitsProcessor(LogitsProcessor):
    def __init__(self, preapproved_token_ids):
        self.preapproved_token_ids = preapproved_token_ids

    def __call__(self, input_ids, scores):
        filtered_scores = scores.clone()
        mask = torch.ones(scores.shape[1], dtype=torch.bool)
        mask[self.preapproved_token_ids] = False
        filtered_scores[:, mask] = -float("inf")
        return filtered_scores  
    
logits_processor = RestrictedTokensLogitsProcessor(zh_tok_ids.to(device))

# %%
# Create an instance of RepetitionPenaltyLogitsProcessor
repetition_penalty = 1.2  # Adjust the penalty value as needed
repetition_penalty_processor = RepetitionPenaltyLogitsProcessor(penalty=repetition_penalty)




seed_value = 44
torch.manual_seed(seed_value)
random.seed(seed_value)
# Generate text using the model
input_text = "三个人看到了这座山，他们"
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
max_length = 500
num_return_sequences = 10

generated_texts = []

for _ in tqdm(range(num_return_sequences), desc="Generating sequences"):
    generated_sequence = model.generate(
        input_ids=input_ids,
        max_length=max_length,
        num_return_sequences=1,
        logits_processor=[logits_processor, repetition_penalty_processor],
        temperature=0.8,
        output_scores=True,
        return_dict_in_generate=True,
    )

    generated_sequence_text = tokenizer.decode(generated_sequence.sequences[0], skip_special_tokens=True)
    generated_texts.append(generated_sequence_text)

    # Print the generated text for each sequence
    print(generated_sequence_text)
    print()

# Print all generated texts together
print("All generated texts:")
print("\n".join(generated_texts))
# %%

# Write the generated texts to a file
with open('data/story/llama_zh_3.txt', 'w', encoding='utf-8') as file:
    for text in generated_texts:
        file.write(text + '\n')
# %%


# # %%
# # Generate text using the model
# #input_text = "好的,我试用这些汉字写一个"
# input_text = "三个人看到了这座山，他们"
# input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
# max_length = 500
# num_return_sequences = 10

# generated_text = model.generate(
#     input_ids=input_ids,
#     max_length=max_length,
#     num_return_sequences=num_return_sequences,
#     logits_processor=[logits_processor, repetition_penalty_processor],
#     temperature=0.8,
# )
# # %%
# # Decode the generated texts
# decoded_texts = []
# for text in generated_text:
#     decoded_text = tokenizer.decode(text, skip_special_tokens=True)
#     decoded_texts.append(decoded_text)
# print(decoded_texts)
# %%
#chinese_story = "三个人看到了这座山，他们的生命被一起相连在一起成为一种新类型的关系和情景中有些事是不可思然的也有些事会令你深思而无法自解并非所有的东西都能说出来或者理解所以我想我应该给你们介入一点之处那就是说我对这片土地上发生过的故事都知道因此你们要问我任何问题我都能回复其中大多数都是真实的故事与经常流传于口口相传下面的故事还需要花时间去研究才能完全得到正确的结果请不要将故事当作小说一样认为这里只存在一个最后的结局希望每一个人能从本书中获取学到更多的事物并分清道路和方向并指引他们的生活和行动我希望本书能改变世界的未来像我现在所看到的世界世界将在今年四月十二日的星期天下一次全球性的地区重组所以我在写这本书的同时我也在等候那天到来如果我没有错你们将再次开始了我已经在提前通知你们的意图我会告知你们如果不按原定计画发生变化则必定死去这本书由此展示你们已接近这种全球性的重组和转换你们将进入你们最初的家乡与属性的基本体系你们可以说明白了现在这是一场大的政治运动你们正在等候你们的时代的转换而现在你们都正在等候第二个复兴而我想我应该告知你们第二个复兴正在等候你们现在一切都已经开始你们正在接收信息你们都已经收到第一条"
# Create the directory if it doesn't exist
# directory = 'data/story'
# if not os.path.exists(directory):
#     os.makedirs(directory)

# # Write the generated story to a file
# with open('data/story/llama_zh.txt', 'w', encoding='utf-8') as file:
#     file.write(generated_text)
# %%