import torch

@torch.no_grad
def logit_lens(prompts, model, tokenizer, only_last_token=True):
    """
    Compute the logits for each layer of a neural network model given a set of prompts.

    Args:
        nn_model (torch.nn.Module): The neural network model.
        prompts (list[str]): The list of prompts.
        only_last_token (bool, optional): Whether to consider only the last token of each prompt. 
            Defaults to True.

    Returns:
        torch.Tensor: The logits per layer of the model.

    """
    model.eval()
    tok_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
    # Todo?: This is a hacky way to get the last token index for each prompt
    last_token_index = tok_prompts.attention_mask.cumsum(1).argmax(-1)
    
    output, cache = model.run_with_cache(prompts) #Expensive!
    
    hidden_l = []
    
    for i in range(model.cfg.n_layers):
        layer_cache = cache[f'blocks.{i}.hook_resid_post']  # (batch, seq, d_model)
        if only_last_token:
            layer_cache = eindex(layer_cache, last_token_index, "i [i] j") # (batch, d_model)
        hidden_l.append(layer_cache) # (batch, seq?, d_model)
            
    hidden = torch.stack(hidden_l, dim=1)  # (batch, num_layers, seq?, d_model)
    rms_out_ln = model.ln_final(hidden) # (batch, num_layers, seq?, d_model)
    logits_per_layer = model.unembed(rms_out_ln) # (batch, num_layers, seq?, vocab_size)
    
    return logits_per_layer


def measure_top_k_accuracy(input_strings, model, tokenizer, top_k_values = [1,5,10]):
    device = next(model.parameters()).device
    tokenized_texts = tokenizer(input_strings, return_tensors="pt").to(device)
    assert tokenized_texts.attention_mask.all()
    output = model(tokenized_texts.input_ids)
    
    test_to_predict = tokenized_texts.input_ids[:, 1:] # (batch, seq-1)
    predictions = output[:, :-1].softmax(dim=-1) # (batch, seq-1, vocab)
    
    accuracies = []  # List to store the accuracies
    for k in top_k_values:
        # Get the top-k indices of the highest values along the last dimension
        top_k_indices = torch.topk(predictions, k, dim=-1).indices # (batch, seq-1, k)
        matches = torch.any(top_k_indices == test_to_predict.unsqueeze(-1), dim=-1)
        accuracy = matches.float().mean().item()
        accuracies.append(accuracy)
        print(f"Top-{k} accuracy: {accuracy * 100}%")
    
    return accuracies

def measure_top_k_accuracy_batched(input_strings, model, tokenizer, top_k_values=[1, 5, 10], batch_size=4):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenized_texts = tokenizer(input_strings, return_tensors="pt", padding=True, truncation=True)
    dataset = torch.utils.data.TensorDataset(tokenized_texts.input_ids, tokenized_texts.attention_mask)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)
    
    accuracies = []
    device = next(model.parameters()).device
    
    for k in top_k_values:
        total_matches = 0
        total_tokens = 0
        
        for batch in dataloader:
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            
            output = model(input_ids, attention_mask=attention_mask)
            
            batch_size = input_ids.size(0)
            seq_length = input_ids.size(1)
            
            test_to_predict = input_ids[:, 1:].contiguous().view(-1)  # (batch * (seq-1))
            predictions = output[:, :-1].contiguous().view(-1, output.size(-1)).softmax(dim=-1)  # (batch * (seq-1), vocab)
            attention_mask = attention_mask[:, 1:].contiguous().view(-1)  # (batch * (seq-1))
            
            top_k_indices = torch.topk(predictions, k, dim=-1).indices  # (batch * (seq-1), k)
            matches = (top_k_indices == test_to_predict.unsqueeze(-1)).any(dim=-1)  # (batch * (seq-1))
            matches = matches.masked_select(attention_mask.bool())  # (num_non_padded)
            
            total_matches += matches.sum().item()
            total_tokens += attention_mask.sum().item()
        
        accuracy = total_matches / total_tokens
        accuracies.append(accuracy)
        print(f"Top-{k} accuracy: {accuracy * 100}%")
    
    return accuracies