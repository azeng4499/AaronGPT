# This implementation is adapted from the GPT model in
# "Build a Large Language Model from Scratch" by Sebastian Raschka.
# Some parts have been modified or extended to better align with the design and 
# functionality requirements of the Better Threads Project.

import time
import torch
import tiktoken
from torch.utils.data import DataLoader
from pretrained_gpt2.create_gpt2_model import create_gpt2_model
from cyberbullying_detector.utils.cb_detect_train_classifer import train_classifier
from cyberbullying_detector.utils.cb_detect_datasets import CyberbullyingDataset
from global_utils import plot_values, log_message

DETECTOR_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 512,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

NUM_WORKERS = 4
BATCH_SIZE = 64
NUM_EPOCHS = 3
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0

def generateChart(name, train_data, val_data, examples_seen, label = 'loss'):
    loss_epochs_tensor = torch.linspace(0, NUM_EPOCHS, len(train_data))
    loss_examples_seen_tensor = torch.linspace(0, examples_seen, len(train_data))

    plot_values(
        loss_epochs_tensor,
        loss_examples_seen_tensor,
        train_data, 
        val_data,
        name,
        label=label
    )

def cb_detect_train(train_data_path, val_data_path):

    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = CyberbullyingDataset(
        csv_file=train_data_path,
        max_length=DETECTOR_CONFIG_124M.context_length,
        tokenizer=tokenizer
    )
    
    val_dataset = CyberbullyingDataset(
        csv_file=val_data_path,
        max_length=DETECTOR_CONFIG_124M.context_length,
        tokenizer=tokenizer
    )

    train_loader = DataLoader(
        dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS, pin_memory=True, drop_last=False,
    )

    model, _, _ = create_gpt2_model(DETECTOR_CONFIG_124M)
    model.to(device)

    log_message(f"GPT2_SMALL Model loaded. Starting training...")

    start_time = time.time()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=NUM_EPOCHS, eta_min=1e-6
    )

    train_losses, val_losses, train_accs, val_accs, examples_seen = \
        train_classifier(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            device=device,
            num_epochs=NUM_EPOCHS,
            eval_freq=100,
            eval_iter=10,
            scheduler=scheduler,
            grad_clip=GRAD_CLIP,
        )

    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    log_message(f"Training completed in {execution_time_minutes:.2f} minutes.")

    torch.save(model.state_dict(), "final_trained_model.pt")

    generateChart(
        name="loss-plot.png", train_data=train_losses, 
        val_data=val_losses, examples_seen=examples_seen
    )

    generateChart(
        name="accuracy-plot.png", train_data=train_accs, 
        val_data=val_accs, examples_seen=examples_seen,
        label="accuracy"
    )