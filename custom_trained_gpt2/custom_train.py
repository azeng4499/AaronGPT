from data_prep.data_prep import create_dataloader
from utils.utils import get_cosine_schedule_with_warmup, log_message, create_training_split, train_model_simple
import torch
import tiktoken
from gpt.gpt import AaronGPTModel


def custom_train(GPT_CONFIG_124M):
    log_message("################## New train ##################")
    start_context = "In the beginning,"
    eval_freq = 100
    eval_iter = 50
    batch_size = 8
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
    total_steps = len(train_loader) * num_epochs
    warmup_steps = int(0.1 * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

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

    train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=eval_freq, eval_iter=eval_iter,
        start_context=start_context, tokenizer=tokenizer, scheduler=scheduler
    )

    torch.save(model.state_dict(), "final_trained_model.pt")