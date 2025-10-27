import torch
import tiktoken
from pretrained_gpt2.create_gpt2_model import create_gpt2_model
from cyberbullying_detector.utils.cb_detect_run import cb_detect_run
from cyberbullying_detector.cb_detect_train import cb_detect_train
from gpt.gpt import AaronGPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True,
    "num_classes": 2
}

def train():
    cb_detect_train(
        "./cyberbullying_detector/data/cyberbullying_train.csv", 
        "./cyberbullying_detector/data/cyberbullying_validation.csv"
    )

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AaronGPTModel(GPT_CONFIG_124M).to(device)
    state_dict = torch.load("final_trained_model.pt", map_location=device)
    model.load_state_dict(state_dict)
    tokenizer = tiktoken.get_encoding("gpt2")

    while True:
        user_input = input("> ")
        if user_input.lower() == "quit":
            break
        print(
            cb_detect_run(
                user_input, model, tokenizer, device, 
                max_length=GPT_CONFIG_124M["context_length"]
            )
        )

def main():
    train()

if __name__ == "__main__":
    main()
