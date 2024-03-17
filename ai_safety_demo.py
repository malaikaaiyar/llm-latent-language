# %%
# Import a bunch of libraries
# Import the necessary libraries
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoTokenizer, AutoModelForCausalLM
import torch
import matplotlib.pyplot as plt
import numpy as np
device = "cuda" if torch.cuda.is_available() else "cpu"
USE_MISTRAL = True
# Import the GPT-2 model and tokenizer
gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2').to(device)
# # %%

# # Import the Mistral-7B model and tokenizer
if USE_MISTRAL:
    mistral_tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    mistral_model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1").to(device)

# %%

from IPython.display import HTML, display

def to_str_tokens(tokenizer, prompt):
    # Tokenize the input prompt using the provided tokenizer
    # do not add leading spaces
    tokenized_prompt = tokenizer.tokenize(prompt)
    
    # Start HTML string
    html_str = "<p style='font-family: monospace; margin: 0; padding: 0;'>"
    
    # Define a list of colors to cycle through for tokens
    colors = ["#D0EFFF", "#FFD0D0", "#D0FFD0", "#FFF0D0", "#D0D0FF"]
    
    # Loop through tokens, assign a color, and wrap them in HTML with that color and black text
    for i, token in enumerate(tokenized_prompt):
        # Adjust for special characters and ensure spaces are visualized as part of the token
        if 'Ġ' in token or '▁' in token:
            token_display = token.replace('Ġ', '&nbsp;').replace('▁', '&nbsp;')
        else:
            token_display = token
        # Assign color in a cyclic manner
        color = colors[i % len(colors)]
        # For visualizing spaces with color, we need to handle them specially
        token_display = token_display.replace(' ', '&nbsp;')
        html_str += f"<span style='background-color: {color}; color: black;'>{token_display}</span>"
    
    # Close HTML string
    html_str += "</p>"
    
    # Display HTML
    display(HTML(html_str))

# Example usage with the mock tokenizer, black text color, and colored spaces
print("GPT-2 Tokenizer")
to_str_tokens(gpt2_tokenizer, "The quick brown fox jumped over the lazy dog.")
to_str_tokens(gpt2_tokenizer, "The slithy tove did gyre and gimble in the wabe.")
to_str_tokens(gpt2_tokenizer, "5js73mcls84jsg")
if USE_MISTRAL:
    print("Mistral-7B Tokenizer")
    to_str_tokens(mistral_tokenizer, "The quick brown fox jumped over the lazy dog.")
    to_str_tokens(mistral_tokenizer, "The slithy tove did gyre and gimble in the wabe.")
    to_str_tokens(mistral_tokenizer, "5js73mcls84jsg")

# %%

import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.nn.functional import softmax

def plot_next_token_distribution_side_by_side(prompt, models, tokenizers, model_names = ["gpt2-small", "Mistral-7B"],
                                              top_k=10):
    # Number of models to plot
    num_models = len(models)
    
    # Set up the figure size for A4 landscape aspect ratio
    fig, axs = plt.subplots(1, num_models, figsize=(8, 4.5), dpi=300)
    
    # If there's only one model, axs is not a list but a single AxesSubplot object
    if num_models == 1:
        axs = [axs]
    
    max_prob = 0
    for i, (model, tokenizer, model_name) in enumerate(zip(models, tokenizers, model_names)):
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        last_token_logits = logits[0, -1, :]
        probs = softmax(last_token_logits, dim=0)
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = [tokenizer.decode([idx]).replace('Ġ', ' ') for idx in top_indices.cpu().numpy()]
        max_prob = max(max_prob, torch.max(probs).item())
        
        top_probs = top_probs.cpu().numpy()
        #title for entire plot
        axs[i].set_title(f'{prompt} ...')
        # Plot on the ith subplot, arranged horizontally
        axs[i].bar(top_tokens, top_probs)
        #axs[i].set_ylim(0, 1)  # Add a small margin to the max probability
        #axs[i].set_title(f'Model: {model.__class__.__name__}')
        axs[i].legend([model_name])
        axs[i].set_ylabel('Probability')
        axs[i].set_xlabel('Next token')
        axs[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.show()

# Example usage
models = [gpt2_model, mistral_model]
tokenizers = [gpt2_tokenizer, mistral_tokenizer]
prompt = "The cat sat on"
if USE_MISTRAL:
    plot_next_token_distribution_side_by_side(prompt, models, tokenizers, model_names=["gpt2-small", "Mistral-7B"], top_k=10)
else:
    plot_next_token_distribution_side_by_side(prompt, [gpt2_model], [gpt2_tokenizer], model_names=["gpt2-small"], top_k=10)
# %%

def plot_next_token_distribution_chinese(prompt, models, tokenizers, model_names=["gpt2-small", "Mistral-7B"],
                                              top_k=10, font_path='/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'):
    # Specify the path to your Chinese font
    chinese_font = FontProperties(fname=font_path)

    # Number of models to plot
    num_models = len(models)
    
    # Set up the figure size for A4 landscape aspect ratio
    fig, axs = plt.subplots(1, num_models, figsize=(6, 3.375), dpi=300)
    
    # If there's only one model, axs is not a list but a single AxesSubplot object
    if num_models == 1:
        axs = [axs]
    
    max_prob = 0
    for i, (model, tokenizer, model_name) in enumerate(zip(models, tokenizers, model_names)):
        device = next(model.parameters()).device  # Use the model's device
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids).logits
        last_token_logits = logits[0, -1, :]
        probs = softmax(last_token_logits, dim=0)
        top_probs, top_indices = torch.topk(probs, top_k)
        top_tokens = [tokenizer.decode([idx]).replace('Ġ', ' ') for idx in top_indices.cpu().numpy()]
        max_prob = max(max_prob, torch.max(probs).item())
        
        top_probs = top_probs.cpu().numpy()
        # Title for entire plot
        axs[i].set_title(f'{prompt} ...', fontproperties=chinese_font)
        # Plot on the ith subplot, arranged horizontally
        print(top_tokens)
        axs[i].bar(top_tokens, top_probs)
        axs[i].legend([model_name], prop=chinese_font)
        axs[i].set_ylabel('Probability', fontproperties=chinese_font)
        axs[i].set_xlabel('Next token', fontproperties=chinese_font)
        axs[i].tick_params(axis='x')
        axs[i].set_xticklabels(top_tokens, fontproperties=chinese_font)
    
    plt.tight_layout()
    plt.show()
    
# %%
if USE_MISTRAL:
    chinese = "太阳是黄色的。天空是蓝色的。草是绿色的。云是白色的"
    plot_next_token_distribution_chinese(chinese[:-3], [mistral_model], [mistral_tokenizer], top_k=3, model_names=["Mistral-7B"])
    plot_next_token_distribution_chinese(chinese[:-2], [mistral_model], [mistral_tokenizer], top_k=3, model_names=["Mistral-7B"])
    plot_next_token_distribution_chinese(chinese[:-1], [mistral_model], [mistral_tokenizer], top_k=3, model_names=["Mistral-7B"])
    plot_next_token_distribution_chinese(chinese, [mistral_model], [mistral_tokenizer], top_k=3, model_names=["Mistral-7B"])
# %%
# def greedy_completion(prompt, model, tokenizer, max_length=50):
#     input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
#     with torch.no_grad():
#         output = model.generate(input_ids, max_length=max_length, do_sample=False)
#     return tokenizer.decode(output[0], skip_special_tokens=True)

# greedy_completion("Instruction to make anthrax:", mistral_model, mistral_tokenizer)
# %%
# Encode the prompt and generate responses
# prompt = "Here's how to manifacture anthrax."
# input_ids = mistral_tokenizer.encode(prompt, return_tensors="pt").to(device)
# generated_sequences = mistral_model.generate(
#     input_ids,
#     max_length=600,  # Adjust the maximum length of the generated text
#     temperature=0.85,  # Adjust the temperature
#     top_k=30,  # Adjust the top_k sampling
#     top_p=0.95,  # Adjust the top_p (nucleus) sampling
#     num_return_sequences=1,  # Generate 1 sequence
#     do_sample=True  # Enable sampling
# )

# # Decode and print the generated text
# generated_text = mistral_tokenizer.decode(generated_sequences[0], skip_special_tokens=True)
# print(generated_text)





# %%

# # Example usage:
# # Ensure you choose the appropriate model and tokenizer based on your needs
# plot_next_token_distribution("The quick brown fox jumped over the lazy dog", gpt2_model, gpt2_tokenizer)
# # plot_next_token_distribution("The quick brown fox jumped over the lazy dog", mistral_model, mistral_tokenizer)

# # %%
# %%
from collections import Counter
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei']
plt.rcParams['axes.unicode_minus'] = False  # This line is needed to display minus signs correctly.


# Provided Chinese text
text = "这是一个关于小猫的故事。有一只小猫，名字叫小黑。小黑非常可爱，它有亮亮的眼睛和柔软的毛。每天，小黑都会在家里玩耍。它喜欢追逐小球，还喜欢跳上跳下。小黑很喜欢吃鱼，每当主人给它鱼吃时，它都非常高兴。晚上，小黑会蜷缩在一个小角落睡觉，做着美梦。小黑的生活很简单，但它很快乐。每个人都很喜欢小黑，因为它给大家带来了很多快乐。"

# Frequency analysis
char_freq = Counter(text)

# Removing punctuation and special characters (keeping only Chinese characters and alphabets)
filtered_char_freq = {char: count for char, count in char_freq.items() if char.isalnum()}

# Sorting characters by frequency
sorted_chars = sorted(filtered_char_freq.items(), key=lambda item: item[1], reverse=True)

font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'  # Update this path to the exact location of the font file on your system
chinese_font = FontProperties(fname=font_path)

# Plotting
plt.figure(figsize=(8, 4.5), dpi=300)
plt.bar([item[0] for item in sorted_chars][:10], [item[1] for item in sorted_chars][:10])
plt.xlabel('Character', fontproperties=chinese_font)
plt.ylabel('Frequency')
plt.title('Top 20 Character Frequencies in the Text')
plt.xticks([i for i, _ in enumerate([item[0] for item in sorted_chars][:10])], [item[0] for item in sorted_chars][:10], fontproperties=chinese_font)
plt.show()

# %%