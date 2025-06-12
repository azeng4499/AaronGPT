import torch
import tiktoken
import re
from datetime import datetime
from data_prep.data_prep import create_dataloader
from gpt.gpt import AaronGPTModel
from config.config import GPT_CONFIG as config
from utils.utils import evaluate_model, log_message, generate_and_print_sample, calc_loss_batch
from data_prep.data_prep import format_data

def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1
    last_checkpoint_path = None

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
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                log_message(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        epoch_train_loss, epoch_val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
        generate_and_print_sample(model, tokenizer, device, start_context)

        checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pt"

        if last_checkpoint_path and os.path.exists(last_checkpoint_path):
            os.remove(last_checkpoint_path)

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_train_loss,
            'val_loss': epoch_val_loss,
        }, checkpoint_path)

        last_checkpoint_path = checkpoint_path 

    return train_losses, val_losses, track_tokens_seen


def __main__():
    with open("./data/test_file.txt", "r", encoding="utf-8") as file:
        text_data = format_data(file)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AaronGPTModel()
        tokenizer = tiktoken.get_encoding("gpt2")
        train_ratio = 0.90
        split_idx = int(train_ratio * len(text_data))
        train_data = text_data[:split_idx]
        val_data = text_data[split_idx:]

        train_loader = create_dataloader(
            train_data,
            batch_size=6,
            max_length=config["context_length"],
            stride=config["context_length"],
            drop_last=True,
            shuffle=True,
            num_workers=0
        )

        val_loader = create_dataloader(
            val_data,
            batch_size=6,
            max_length=config["context_length"],
            stride=config["context_length"],
            drop_last=False,
            shuffle=False,
            num_workers=0
        )

        model.to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=0.0004, weight_decay=0.1
        )
        num_epochs = 10
        log_message("################## New train ##################")

        train_losses, val_losses, track_tokens_seen = train_model(
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

__main__()



