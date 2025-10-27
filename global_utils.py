from datetime import datetime
import matplotlib.pyplot as plt

def log_message(msg):
    if len(msg) > 0:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("/root/AaronGPT/logs/training_logs.txt", "a") as f:
            f.write(f"{timestamp} -> {msg}\n")

def plot_values(
    epochs_seen, examples_seen, train_values, val_values, save_file_path, label="loss"
):
    fig, ax1 = plt.subplots(figsize=(5, 3))

    ax1.plot(epochs_seen, train_values, label=f"Training {label}")
    ax1.plot(
        epochs_seen, val_values, linestyle="-.", 
        label=f"Validation {label}"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel(label.capitalize())
    ax1.legend()

    ax2 = ax1.twiny()
    ax2.plot(examples_seen, train_values, alpha=0)
    ax2.set_xlabel("Examples seen")

    fig.tight_layout()
    plt.savefig(f"{label}-plot.pdf")
    plt.savefig(save_file_path)

