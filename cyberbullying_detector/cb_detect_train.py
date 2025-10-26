import time
import torch
import tiktoken
from torch.utils.data import DataLoader
from pretrained_gpt2.create_gpt2_model import create_gpt2_model
from cyberbullying_detector.utils.cb_detect_train_classifer import train_classifier
from cyberbullying_detector.utils.cb_detect_datasets import CyberbullyingDataset
from global_utils import plot_values, log_message

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}


def cb_detect_train():

    num_workers = 8
    batch_size = 32
    num_epochs = 3
    learning_rate = 1e-5
    weight_decay = 0.01
    grad_clip = 1.0

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CyberbullyingDataset(
        csv_file="cyberbullying_detector/data/cyberbullying_train.csv",
        max_length=128,
        tokenizer=tokenizer
    )
    
    val_dataset = CyberbullyingDataset(
        csv_file="cyberbullying_detector/data/cyberbullying_validation.csv",
        max_length=train_dataset.max_length,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=batch_size,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )

    model, settings, params = create_gpt2_model(GPT_CONFIG_124M)
    model.to(device)

    start_time = time.time()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs, eta_min=1e-6
    )

    train_losses, val_losses, train_accs, val_accs, examples_seen = \
        train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=num_epochs,
            eval_freq=100,
            eval_iter=10,
            scheduler=scheduler,
            grad_clip=grad_clip
        )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    log_message(f"Training completed in {execution_time_minutes:.2f} minutes.")

    torch.save(model.state_dict(), "final_trained_model.pt")

    plot_values(
        torch.arange(len(train_losses)),
        torch.arange(examples_seen, steps=len(train_losses)),
        train_losses, 
        val_losses,
        "tv_loss.png"
    )

    plot_values(
        torch.arange(len(train_accs)),
        torch.arange(examples_seen, steps=len(train_accs)),
        train_accs, 
        val_accs,
        "tv_acc.png", 
        label="accuracy"
    )
