# %%
from nnsight import LanguageModel
from dataclasses import dataclass, field
import torch


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

# We'll never actually load the parameters so no need to specify a device_map.
model = LanguageModel("meta-llama/Llama-2-70b-hf")

@torch.no_grad
def logit_lens(nn_model, prompts, idx_family):
    """
    Get the probabilities of the next token for the last token of each prompt at each layer using the logit lens.

    Args:
        nn_model: NNSight LanguageModel object
        prompts: List of prompts or a single prompt
        idx_family: Int tensor shape (num_prompts,) of the desired token indices for each prompt

    Returns:
        A tensor of shape (num_prompts, num_layers, vocab_size) containing the probabilities
        of the next token for each prompt at each layer. Tensor is on the CPU.
    """
    nn_model.eval()
    tok_prompts = nn_model.tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index
    last_token_index = (
        tok_prompts.attention_mask.flip(1).cumsum(1).bool().int().sum(1).sub(1)
    )
    with nn_model.trace(prompts, remote=True, scan=True, validate=True) as tracer:
        hiddens_l = [
            layer.output[0][
                torch.arange(len(tok_prompts.input_ids)),
                last_token_index,
            ].unsqueeze(1)
            for layer in nn_model.model.layers
        ]
        hiddens = torch.cat(hiddens_l, dim=1).save()
        #rms_out = nn_model.model.norm(hiddens)
        #logits = nn_model.lm_head(rms_out).save()
        #probs = logits.softmax(-1)
        #prob_idx = probs[:, :, idx_family].save()
        return hiddens

prompts = ["The cat sat on the mat", "The dog ate the", "The man walked towards the"]
logit_lens(model, prompts, torch.tensor([1, 2, 3]))
# %%