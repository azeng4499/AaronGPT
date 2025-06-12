
import torch
from datetime import datetime

def log_message(msg):
    if len(msg) > 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("training_logs.txt", "a") as f:
            f.write(f"{timestamp} -> {msg}\n")

def text_to_token(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|EOS|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_to_text(tokens, tokenizer):
    flat = tokens.squeeze(0)
    text = tokenizer.decode(flat.tolist())
    return text


def generate_text(model, idx, max_new_tokens, context_size, temperature=1.0, top_k=50):
    model.eval()
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]

        # with torch.no_grad():
        logits = model(idx_cond)
        logits = logits[:, -1, :]  
        logits /= temperature                

        if top_k is not None and top_k > 0:
            top_k_values, _ = torch.topk(logits, top_k)
            min_top_k = top_k_values[:, -1].unsqueeze(-1)
            logits = torch.where(logits < min_top_k, float("-inf"), logits)

        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1) 

    return idx 

def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
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
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches

def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    # with torch.no_grad():
    train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
    val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.pos_embed.weight.shape[0]
    encoded = text_to_token(start_context, tokenizer).to(device)
    # with torch.no_grad():
    token_ids = generate_text(
        model=model, 
        idx=encoded,
        max_new_tokens=50, 
        context_size=context_size,
    )
    decoded_text = token_to_text(token_ids, tokenizer)
    print(decoded_text.replace("\n", " "))
    model.train()