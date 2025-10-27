# This implementation is adapted from the GPT model in
# "Build a Large Language Model from Scratch" by Sebastian Raschka.
# Some parts have been modified or extended to better align with the design and 
# functionality requirements of the Better Threads Project.

import torch

def calc_accuracy_loader(data_loader, model, device, num_batches=None, pad_token_id=50256):
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch = input_batch.to(device)
            target_batch = target_batch.to(device)


            with torch.no_grad():
                hidden_states = model.get_hidden_states(input_batch)
                attention_mask = (input_batch != pad_token_id).float()
                
                masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
                seq_lengths = attention_mask.sum(dim=1, keepdim=True)
                pooled_hidden = masked_hidden.sum(dim=1) / seq_lengths
                
                classification_logits = model.out_head(pooled_hidden)
                predicted_labels = torch.argmax(classification_logits, dim=-1)
        
            num_examples += predicted_labels.shape[0]
            correct_predictions += (
                (predicted_labels == target_batch).sum().item()
            )
        else:
            break

    return correct_predictions / num_examples

def calc_loss_batch(input_batch, target_batch, model, device, pad_token_id=50256):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)


    hidden_states = model.get_hidden_states(input_batch)

    attention_mask = (input_batch != pad_token_id).float() 

    masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
    seq_lengths = attention_mask.sum(dim=1, keepdim=True)
    pooled_hidden = masked_hidden.sum(dim=1) / seq_lengths

    classification_logits = model.out_head(pooled_hidden)

    loss = torch.nn.functional.cross_entropy(classification_logits, target_batch)

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.

    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches
