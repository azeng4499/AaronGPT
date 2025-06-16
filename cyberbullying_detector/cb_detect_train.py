import time
import torch
import tiktoken
import pandas as pd
from gpt.gpt import AaronGPTModel
from torch.utils.data import DataLoader
from pretrained_gpt2.create_gpt2_model import create_gpt2_model
from cyberbullying_detector.utils.cb_detect_train_classifer import train_classifier
from cyberbullying_detector.utils.cb_detect_datasets import CyberbullyingDataset
from global_utils import plot_values, log_message

def cb_detect_train(GPT_CONFIG_124M):

    num_workers = 0
    batch_size = 8
    num_epochs = 10
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CyberbullyingDataset(
        csv_file="cyberbullying_detector/data/cyberbullying_train.csv",
        max_length=None,
        tokenizer=tokenizer
    )
    val_dataset = CyberbullyingDataset(
        csv_file="cyberbullying_detector/data/cyberbullying_validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size,
        num_workers=num_workers, drop_last=False,
    )

    model, settings, params = create_gpt2_model(GPT_CONFIG_124M)

    start_time = time.time()
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=5e-6, 
        weight_decay=1e-4
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    train_losses, val_losses, train_accs, val_accs, examples_seen = \
        train_classifier(
            model, train_loader, val_loader, optimizer, device,
            num_epochs=num_epochs, eval_freq=100,
            eval_iter=5, scheduler=scheduler
        )
    
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    log_message(f"Training completed in {execution_time_minutes:.2f} minutes.")

    torch.save(model.state_dict(), "final_trained_model.pt")

    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))

    plot_values(
        epochs_tensor, 
        examples_seen_tensor, 
        train_losses, 
        val_losses,
        "tv_loss.png"
    )

    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))

    plot_values(
        epochs_tensor, 
        examples_seen_tensor, 
        train_accs, 
        val_accs,
        "tv_acc.png",
        label="accuracy"
    )
