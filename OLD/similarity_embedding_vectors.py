# %%
import torch
from transformer_lens import HookedTransformer
from tqdm import tqdm
torch.set_grad_enabled(False)
from transformers import AutoTokenizer
from utils.misc import plot_ci, lang2name
import matplotlib.pyplot as plt
import json
from dataclasses import dataclass, field
from einops import rearrange
import numpy as np
from sklearn.manifold import TSNE
#disable gradients
torch.set_grad_enabled(False)

@dataclass
class Config:
    source_lang: str = 'zh'
    target_lang: str = 'ko'
    think_lang: str = 'en'
    model_name: str = 'meta-llama/Llama-2-7b-hf'
    word_dict_path: str = 'data/filtered_word_dict.json'
    base_prompt: str = '中文:花 한국어:꽃 中文:山 한국어:산 中文:月 한국어:달 中文:水 한국어:물 '
    model_kwargs: dict = field(default_factory=dict)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

cfg = Config()
cfg.model_kwargs = {'use_fast': False, 'add_prefix_space': False}


device = torch.device('cpu')

if 'model' not in locals():
    model = HookedTransformer.from_pretrained(cfg.model_name, device=device)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    
    
    
tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, **cfg.model_kwargs)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


with open(cfg.word_dict_path, 'r') as f:
    word_dict = json.load(f)
# %%

lang_code = ['en', 'zh', 'ko']
embedding_vectors = torch.zeros((3, len(word_dict), model.cfg.d_model))
names = []
for i, lang in enumerate(lang_code):
    for j, baseword in tqdm(enumerate(word_dict.keys())):
        if baseword == "six":
            continue
        token_id = word_dict[baseword][lang][1]
        embedding_vectors[i, j] = model.embed.W_E[token_id]
        names.append(f"{lang}_{baseword}")
# %%
import plotly.graph_objects as go

all_embed_vec = rearrange(embedding_vectors, 'l w d -> (w l) d').numpy()
languages = np.repeat(np.arange(3), embedding_vectors.shape[1])  # Assuming 3 languages

# tsne = TSNE(n_components=3, random_state=0)
tsne = TSNE(n_components=2, random_state=0)  # Modify to 2D

# reduced_vectors = tsne.fit_transform(all_embed_vec)

# Step 3: Plot the 3D points colored by their language
reduced_vectors = tsne.fit_transform(all_embed_vec)  # Update variable name

fig = go.Figure(data=[go.Scatter(
    x=reduced_vectors[:, 0],
    y=reduced_vectors[:, 1],
    mode='markers+text',  # Add text labels to the markers
    text=names,  # Use the names as the text labels
    textposition='top center',  # Position the text labels on top of the markers
    marker=dict(
        size=5,
        color=languages,  # Color points by language
        colorscale='Viridis',  # Choose a colorscale
        opacity=0.8
    )
)])

fig.update_layout(
    xaxis_title='t-SNE Feature 1',
    yaxis_title='t-SNE Feature 2',
    title='Dimensionality Reduction of Vectors by Language',
    width=800,  # Set the width of the figure
    height=800,  # Set the height of the figure
    autosize=False,  # Disable autosizing
    margin=dict(l=0, r=0, b=0, t=0),  # Set the margin to 0 on all sides
)

fig.show()


# %%
import torch
import torch.nn.functional as F

# Your tensor of shape (3, N, d_model)
# embedding_vectors = torch.zeros((3, N, d_model))  # Example initialization

# Compute cosine similarities
# Reshape the tensor to compute pairwise similarities across the first dimension (languages)
cos_sim_en_zh = F.cosine_similarity(embedding_vectors[0], embedding_vectors[1], dim=1)
cos_sim_en_ko = F.cosine_similarity(embedding_vectors[0], embedding_vectors[2], dim=1)
cos_sim_zh_ko = F.cosine_similarity(embedding_vectors[1], embedding_vectors[2], dim=1)

# Average the cosine similarities over N
avg_sim_en_zh = cos_sim_en_zh.mean().item()
avg_sim_en_ko = cos_sim_en_ko.mean().item()
avg_sim_zh_ko = cos_sim_zh_ko.mean().item()

# Compute the standard error of the mean (SME)
sme_en_zh = cos_sim_en_zh.std().item() / np.sqrt(len(cos_sim_en_zh))
sme_en_ko = cos_sim_en_ko.std().item() / np.sqrt(len(cos_sim_en_ko))
sme_zh_ko = cos_sim_zh_ko.std().item() / np.sqrt(len(cos_sim_zh_ko))

# Print average similarities and SME
print(f"Average similarity between English and Chinese: {avg_sim_en_zh} (SME: {sme_en_zh})")
print(f"Average similarity between English and Korean: {avg_sim_en_ko} (SME: {sme_en_ko})")
print(f"Average similarity between Chinese and Korean: {avg_sim_zh_ko} (SME: {sme_zh_ko})")

# %%

# Concatenate the embedding vectors along the first dimension
concatenated_vectors = torch.cat((embedding_vectors[0], embedding_vectors[1], embedding_vectors[2]), dim=0)

# Plotting the average similarities with error bars
x = ['en-zh', 'en-ko', 'zh-ko']
all_embed_vec = torch.tensor(all_embed_vec)  # Convert numpy array to PyTorch tensor

# Compute cosine similarity between every pair of vectors
import torch 
import torch
def plot_cosine_similarity_matrix(vectors, **kwargs):
    """
    Takes a matrix of shape (N,D), where N is the number of vectors and D is the dimensionality.
    Computes the cosine similarity between every pair of vectors and plots the (N,N) grid of similarities.
    
    Parameters:
    vectors (Tensor): A tensor of shape (N, D) containing N vectors of dimension D.
    """
    # Compute cosine similarity matrix
    cos_sim_matrix = F.cosine_similarity(vectors.unsqueeze(1), vectors.unsqueeze(0), dim=-1)
    
    # Plot the cosine similarity matrix as an image
    fig = go.Figure(data=go.Heatmap(
                       z=cos_sim_matrix.numpy(),
                       colorscale='Viridis'))
    
    fig.update_layout(
        **kwargs,
    )
    
    fig.show()

for i in range(3):
    plot_cosine_similarity_matrix(embedding_vectors[i],
                                  title='Cosine Similarity Matrix',
                                    xaxis_title='Vector Index',
                                    yaxis_title='Vector Index',
                                    width=800,  # Set the width of the figure
                                    height=800,  # Set the height of the figure
                                    autosize=False,  # Disable autosizing
                                    margin=dict(l=0, r=0, b=0, t=0),  # Set the margin to 0 on all sides
                                )
# %%
cos_sim_matrix = F.cosine_similarity(embedding_vectors[2].unsqueeze(1), embedding_vectors[2].unsqueeze(0), dim=-1)
epsilon = 0.8
indices = torch.where((cos_sim_matrix > epsilon) & (torch.eye_like(cos_sim_matrix) == 0))
base_lookup = list(word_dict.keys())
for i,j in zip(*indices):
    wi,wj = word_dict[base_lookup[i]]['ko'], word_dict[base_lookup[j]]['ko']
    print(f"Similar words: {wi} and {wj}")
    print(f"Cosine similarity: {cos_sim_matrix[i,j].item()}")


# %%
import plotly.graph_objects as go

korean_words = [word_dict[word]['ko'][0] for word in word_dict]

# Calculate the norm of the embedding vectors
embedding_norms = torch.norm(all_embed_vec, dim=-1).cpu().numpy()

# Plotting the norm of the embedding vectors as a bar chart
fig = go.Figure(data=[go.Bar(x=list(range(len(embedding_norms))), y=embedding_norms)])
fig.update_layout(
    xaxis_title='Vector Index',
    yaxis_title='Norm',
    title='Norm of Embedding Vectors'
)
fig.show()
# %%
