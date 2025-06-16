import torch
import tiktoken
from pretrained_gpt2.create_gpt2_model import create_gpt2_model
from cyberbullying_detector.utils.cb_detect_run import cb_detect_run
from gpt.gpt import AaronGPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 8,
    "n_layers": 6,
    "drop_rate": 0.1,
    "qkv_bias": True
}

def main():
    #train

    # cb_detect_train(GPT_CONFIG_124M)

    #run

    model, settings, params = create_gpt2_model(GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input = "you are so dumb, you need to die"
    model_state_dict = torch.load("./final_trained_model.pt", map_location=device)
    model.load_state_dict(model_state_dict)

    print(cb_detect_run(
        input, 
        model, 
        tokenizer, 
        device, 
        max_length=GPT_CONFIG_124M["context_length"]
    ))



if __name__ == "__main__":
    main()
