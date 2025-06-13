import torch
import tiktoken
from data_prep.data_prep import create_dataloader
from gpt.gpt import AaronGPTModel
from utils.utils import create_training_split, train_model_simple
from data_prep.data_prep import format_data


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}

def main():
    start_context = "In the beginning,"
    eval_freq = 5
    eval_iter = 5
    batch_size = 2
    num_epochs = 10
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AaronGPTModel(GPT_CONFIG_124M)
    model.to(device)
    text_data = ""
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0004,
        weight_decay=0.1
    )

    try:
        with open("./data/the_verdict.txt", "r", encoding="utf-8") as file:
            text_data = file.read()
    except Exception as e:
        raise RuntimeError(f"Failed to load training data: {e}")
        
    train_data, val_data = create_training_split(0.9, text_data)

    train_loader = create_dataloader(
        train_data, batch_size=batch_size, max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"] // 2, drop_last=True,
        shuffle=True, num_workers=0
    )
    val_loader = create_dataloader(
        val_data, batch_size=batch_size, max_length=GPT_CONFIG_124M["context_length"],
        stride=GPT_CONFIG_124M["context_length"] // 2, drop_last=False, 
        shuffle=False, num_workers=0
    )

    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=start_context, tokenizer=tokenizer
    )

main()
























# def train_model(model, train_loader, val_loader, optimizer, device, num_epochs, eval_freq, eval_iter, start_context, tokenizer, scheduler):
#     early_stopping_patience = 5 
#     best_val_loss = float('inf')
#     patience_counter = 0
    
#     train_losses, val_losses, track_tokens_seen = [], [], []
#     tokens_seen, global_step = 0, -1
#     last_checkpoint_path = None

#     for epoch in range(num_epochs):
#         model.train()
#         for input_batch, target_batch in train_loader:
#             optimizer.zero_grad()
#             loss = calc_loss_batch(input_batch, target_batch, model, device)
#             loss.backward()
#             optimizer.step()
#             tokens_seen += input_batch.numel()
#             global_step += 1
#             if global_step % eval_freq == 0:
#                 train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
#                 train_losses.append(train_loss)
#                 val_losses.append(val_loss)
#                 track_tokens_seen.append(tokens_seen)
#                 log_message(f"Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

#         epoch_train_loss, epoch_val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)
#         generate_and_print_sample(model, tokenizer, device, start_context)

#         checkpoint_path = f"checkpoints/checkpoint_epoch_{epoch + 1}.pt"

#         if last_checkpoint_path and os.path.exists(last_checkpoint_path):
#             os.remove(last_checkpoint_path)

#         torch.save({
#             'epoch': epoch,
#             'model_state_dict': model.state_dict(),
#             'optimizer_state_dict': optimizer.state_dict(),
#             'train_loss': epoch_train_loss,
#             'val_loss': epoch_val_loss,
#         }, checkpoint_path)

#         last_checkpoint_path = checkpoint_path 

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             patience_counter = 0
#         else:
#             patience_counter += 1
            
#         if patience_counter >= early_stopping_patience:
#             print(f"Early stopping at epoch {epoch}")
#             break
            
#         scheduler.step()

#     return train_losses, val_losses, track_tokens_seen


# def __main__():
#     with open("./data/train_data.txt", "r", encoding="utf-8") as file:
#         text_data = format_data(file)

#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         model = AaronGPTModel()
#         tokenizer = tiktoken.get_encoding("gpt2")
#         train_ratio = 0.90
#         split_idx = int(train_ratio * len(text_data))
#         train_data = text_data[:split_idx]
#         val_data = text_data[split_idx:]

#         train_loader = create_dataloader(
#             train_data,
#             batch_size=8,
#             max_length=config["context_length"],
#             stride=config["context_length"] // 2,
#             drop_last=True,
#             shuffle=True,
#             num_workers=0
#         )

#         val_loader = create_dataloader(
#             val_data,
#             batch_size=8,
#             max_length=config["context_length"],
#             stride=config["context_length"] // 2,
#             drop_last=False,
#             shuffle=False,
#             num_workers=0
#         )

#         model.to(device)

#         optimizer = torch.optim.AdamW(
#             model.parameters(),
#             lr=0.0002, 
#             weight_decay=0.1
#         )

#         scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
#             optimizer, 
#             mode='min',           # Monitor for minimum val_loss
#             factor=0.7,           # Reduce LR by 30% when triggered
#             patience=3,           # Wait 3 epochs before reducing
#             verbose=True,         # Print when LR is reduced
#             min_lr=1e-6          # Don't go below this LR
#         )
        
#         num_epochs = 25
#         log_message("################## New train ##################")

#         train_losses, val_losses, track_tokens_seen = train_model(
#             model, 
#             train_loader, 
#             val_loader, 
#             optimizer, 
#             device,
#             num_epochs=num_epochs, 
#             eval_freq=200, 
#             eval_iter=50,
#             start_context="In the future", 
#             tokenizer=tokenizer,
#             scheduler=scheduler
#         )    
#         torch.save(model.state_dict(), "trained_model.pt")

# __main__()



