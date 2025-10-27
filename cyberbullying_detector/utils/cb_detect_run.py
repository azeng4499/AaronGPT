import torch


def cb_detect_run(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    model.to(device)

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_embed.weight.shape[1]
    input_ids = input_ids[:min(max_length, supported_context_length)]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)

    with torch.no_grad():
        hidden_states = model.get_hidden_states(input_tensor)
        attention_mask = (input_tensor != pad_token_id).float()
        
        masked_hidden = hidden_states * attention_mask.unsqueeze(-1)
        seq_lengths = attention_mask.sum(dim=1, keepdim=True)
        pooled_hidden = masked_hidden.sum(dim=1) / seq_lengths
        
        classification_logits = model.out_head(pooled_hidden)
        probs = torch.softmax(classification_logits, dim=-1)
        
        predicted_label = torch.argmax(probs, dim=-1)
        confidence = probs[0, predicted_label].item()

    label = "cyberbullying" if predicted_label == 1 else "not cyberbullying"
    return label, confidence