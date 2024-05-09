# %%
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
torch.set_grad_enabled(False)

# %%
# Set torch device to use CPU only
device = torch.device('cpu')
model = HookedTransformer.from_pretrained("qwen-1.8b", device=device)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
# %%
# Read lines from single_tok_lang.txt
def read_tuples_from_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize empty list for 4-tuples
    tuples = []

    # Process each line and create 4-tuples
    for line in lines:
        # Split the line into four parts
        try:
            en, zh, ko = line.strip().split(',')
            tuples.append((en, zh, ko))
        except:
            print(f'Error with line: {line}')

    return tuples

# %%
# Read lines from single_tok_lang.txt
tuples = read_tuples_from_file('single_tok_lang3.txt')

# Print the 4-tuples
for tuple in tuples:
    print(tuple)

base_prompt = "English:flower 한국어:꽃 English:mountain 한국어:산 English:moon 한국어:달 English:water 한국어:물 "


zh_answer_ids, en_answer_ids, ko_answer_ids, prompts = [], [], [], []

for en, zh, ko in tuples:
    en_tokens, zh_tokens, ko_tokens = model.tokenizer(en)['input_ids'], model.tokenizer(zh)['input_ids'], model.tokenizer(ko)['input_ids']
    
    if len(en_tokens) == 1 and len(zh_tokens) == 1 and len(ko_tokens) == 1:
        en_id, zh_id, ko_id = en_tokens[0], zh_tokens[0], ko_tokens[0]
        en_answer_ids.append(en_id)
        zh_answer_ids.append(zh_id)
        ko_answer_ids.append(ko_id)
        prompt = base_prompt + f"English:{en} 한국어:"
        prompts.append(prompt)

# %%

zh_probs = torch.zeros(model.cfg.n_layers, len(prompts))
ko_probs = torch.zeros(model.cfg.n_layers, len(prompts))

for i, prompt in tqdm(enumerate(prompts)):
    output, cache = model.run_with_cache(prompt)
    
    for j in range(model.cfg.n_layers):
        resid = cache[f'blocks.{j}.hook_resid_post'] 
        ln_resid = model.ln_final(resid)
        logits = model.unembed(ln_resid)
        #logits = ln_resid[0, -1, :] @ model.unembed.W_U + model.unembed.b_U
        zh_prob = torch.softmax(logits[0, -1, :], dim=-1)[zh_answer_ids[i]].item()
        ko_prob = torch.softmax(logits[0, -1, :], dim=-1)[ko_answer_ids[i]].item()
        
        zh_probs[j, i] = zh_prob
        ko_probs[j, i] = ko_prob
        
zh_probs_mean = zh_probs.mean(dim=1)
zh_probs_std = zh_probs.std(dim=1)
ko_probs_mean = ko_probs.mean(dim=1)
ko_probs_std = ko_probs.std(dim=1)

# %%
import matplotlib.pyplot as plt

# Plotting en_probs_mean with envelope
plt.plot(range(model.cfg.n_layers), zh_probs_mean, label='Chinese')
plt.fill_between(range(model.cfg.n_layers), zh_probs_mean - zh_probs_std, zh_probs_mean + zh_probs_std, alpha=0.3)

# Plotting ko_probs_mean with envelope
plt.plot(range(model.cfg.n_layers), ko_probs_mean, label='Korean')
plt.fill_between(range(model.cfg.n_layers), ko_probs_mean - ko_probs_std, ko_probs_mean + ko_probs_std, alpha=0.3)

# Set plot title and labels
plt.title('Language Probs')
plt.xlabel('Layer')
plt.ylabel('Probability')

# Add legend
plt.legend()

# Show the plot
plt.show()
# %%
