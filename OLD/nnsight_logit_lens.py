# %%
from nnsight import LanguageModel
from nnsight import CONFIG
import os
import torch

HF_AUTH = "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"
os.environ["HF_TOKEN"] = HF_AUTH

NNSIGHT_AUTH = "7SeBM7tCoyXvyjG4KjGb"
CONFIG.set_default_api_key(NNSIGHT_AUTH)

# !pip uninstall -y transformers
# !pip install git+https://github.com/huggingface/transformers

# %%

model = LanguageModel("meta-llama/Llama-2-70b-hf", dispatch=True)
torch.set_grad_enabled(False)
# %%
# All we need to specify using NDI

logit_lens(model, prompts)
# %%
        
    
# %%
def latent_lang(
    prompts: List[str], 
    model: Any, 
    id_family: List[Tuple[int, int]]) -> Float[Tensor, "n_langs n_layers n_prompts"]:
    """
    Perform latent language modeling on the given prompts using the provided model.

    Args:
        prompts (str): The prompts to generate language models for.
        model: The language model to use for generation.
        id_family: The family of IDs for each language.

    Returns:
        layer_probs (torch.Tensor): The probabilities of the next token for each language, layer, and prompt.
    """
    n_langs = len(id_family)
    layer_probs = torch.zeros_like((n_langs, len(prompts), model.config.n_layers))
    for i in tqdm(range(len(prompts))):
        output, cache = model.run_with_cache(prompts[i])
        for j in range(model.config.n_layers):
            resid = cache[f'blocks.{j}.hook_resid_post'] 
            ln_resid = model.ln_final(resid)
            logits = model.unembed(ln_resid)
            next_tok_prob = torch.softmax(logits[0, -1], dim=-1)
            for k in range(n_langs):
                layer_probs[k,j,i] = next_tok_prob[id_family[k,i]].sum()
    return layer_probs


model.trace