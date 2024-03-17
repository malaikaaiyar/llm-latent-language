# %%
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
torch.set_grad_enabled(False)

# %%
# Set torch device to use CPU only
device = torch.device('cpu')
model = HookedTransformer.from_pretrained("qwen-7b", device=device)

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
            en, zh, ko, es = line.strip().split(',')
            tuples.append((en, zh, ko, es))
        except:
            print(f'Error with line: {line}')

    return tuples

# %%
# Read lines from single_tok_lang.txt
tuples = read_tuples_from_file('simple_tok_en.txt')

# Print the 4-tuples
for tuple in tuples:
    print(tuple)
# %%
#base_prompt = "English:flower 한국어:꽃 English:mountain 한국어:산 English:moon 한국어:달 English:water 한국어:물 "
base_prompt = "English:flower Español:flor English:mountain Español:montaña English:moon Español:luna English:water Español:agua"






zh_answer_ids, en_answer_ids, ko_answer_ids, es_answer_ids , prompts = [], [], [], [], []

for en, zh, ko, es in tuples:
    en_tokens, zh_tokens, es_tokens = model.tokenizer(en)['input_ids'], model.tokenizer(zh)['input_ids'], model.tokenizer(es)['input_ids']
    
    if len(en_tokens) == 1 and len(zh_tokens) == 1 and len(es_tokens) == 1:
        en_id, zh_id, es_id = en_tokens[0], zh_tokens[0], es_tokens[0]
        en_answer_ids.append(en_id)
        zh_answer_ids.append(zh_id)
        es_answer_ids.append(es_id)
        prompt = base_prompt + f"English:{en} Español:"
        prompts.append(prompt)

# %%

zh_probs = torch.zeros(model.cfg.n_layers, len(prompts))
es_probs = torch.zeros(model.cfg.n_layers, len(prompts))

for i, prompt in tqdm(enumerate(prompts)):
    output, cache = model.run_with_cache(prompt)
    
    for j in range(model.cfg.n_layers):
        resid = cache[f'blocks.{j}.hook_resid_post'] 
        ln_resid = model.ln_final(resid)
        logits = model.unembed(ln_resid)
        #logits = ln_resid[0, -1, :] @ model.unembed.W_U + model.unembed.b_U
        zh_prob = torch.softmax(logits[0, -1, :], dim=-1)[zh_answer_ids[i]].item()
        es_prob = torch.softmax(logits[0, -1, :], dim=-1)[es_answer_ids[i]].item()
        
        zh_probs[j, i] = zh_prob
        es_probs[j, i] = es_prob
        
zh_probs_mean = zh_probs.mean(dim=1)
zh_probs_std = zh_probs.std(dim=1)
es_probs_mean = es_probs.mean(dim=1)
es_probs_std = es_probs.std(dim=1)

# %%
import matplotlib.pyplot as plt

# Plotting en_probs_mean with envelope
plt.plot(range(model.cfg.n_layers), zh_probs_mean, label='Chinese')
plt.fill_between(range(model.cfg.n_layers), zh_probs_mean - zh_probs_std, zh_probs_mean + zh_probs_std, alpha=0.3)

# Plotting es_probs_mean with envelope
plt.plot(range(model.cfg.n_layers), es_probs_mean, label='Spanish')
plt.fill_between(range(model.cfg.n_layers), es_probs_mean - es_probs_std, es_probs_mean + es_probs_std, alpha=0.3)

# Set plot title and labels
plt.title('Language Probs EN -> ES')
plt.xlabel('Layer')
plt.ylabel('Probability')

# Add legend
plt.legend()

# Show the plot
plt.show()
# %%
# Plotting each line in zh_probs with different colors
for i in range(len(prompts)):
    plt.plot(range(model.cfg.n_layers), zh_probs[:, i], label=f'Chinese {i+1}', color='blue')
    plt.plot(range(model.cfg.n_layers), es_probs[:, i], label=f'Spanish {i+1}', color='orange')

# Set plot title and labels
plt.title('Language Probs EN -> ES')
plt.xlabel('Layer')
plt.ylabel('Probability')

# Add legend
plt.legend()

# Show the plot
plt.show()

# %%
