import torch
import tiktoken
from datetime import datetime
from data_prep.data_prep import create_dataloader
from gpt.gpt import AaronGPTModel
from config.config import GPT_CONFIG as config
from utils.utils import evaluate_model, log_message, generate_and_print_sample, generate_text, calc_loss_batch


def train_model(model, train_loader, val_loader,
    optimizer, device, num_epochs,
    eval_freq, eval_iter, start_context, tokenizer):
    # train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
                # train_losses.append(train_loss)
                # val_losses.append(val_loss)
                # track_tokens_seen.append(tokens_seen)
                log_message(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        generate_and_print_sample(model, tokenizer, device, start_context)
    # return train_losses, val_losses, track_tokens_seen

def __main__():
    with open("./utils/data/the_verdict.txt", "r", encoding="utf-8") as file:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        text_data = file.read()
        model = AaronGPTModel()
        tokenizer = tiktoken.get_encoding("gpt2")
        train_ratio = 0.90
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]

        train_loader = create_dataloader(
            train_data,
            batch_size=2,
            max_length=config["context_length"],
            stride=config["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0
        )
        val_loader = create_dataloader(
            val_data,
            batch_size=2,
            max_length=config["context_length"],
            stride=config["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )

        model = AaronGPTModel()
        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0004, weight_decay=0.1
        )
        num_epochs = 15
        log_message("\nNew train - " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        train_model(
            model, 
            train_loader, 
            val_loader, 
            optimizer, 
            device,
            num_epochs=num_epochs, 
            eval_freq=5, 
            eval_iter=5,
            start_context="The universe does", 
            tokenizer=tokenizer
        )    

        torch.save(model.state_dict(), "trained_model.pt")

def model_test(
    input_text,
    model_path="trained_model.pt",
    max_new_tokens=50,
    temperature=1.0,
    top_k=40,
    context_size=1024
):
    if not input_text or not isinstance(input_text, str):
        raise ValueError("You must provide an input text.")

    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = tokenizer.encode(input_text, allowed_special={'<|EOS|>'})
    input_ids = torch.tensor([input_ids])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AaronGPTModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    output_ids = generate_text(
        model=model,
        idx=input_ids.to(device),
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
        tokenizer=tokenizer
    )

    output_text = token_to_text(output_ids, tokenizer)
    print("Generated text:\n", output_text)
    return output_ids



model_test("In the depths of the forest,")
# __main__()



