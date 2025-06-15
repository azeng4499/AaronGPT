import torch
import tiktoken
from pretrained_gpt2.create_pretrained_gpt2 import create_pretrained_gpt2_model
from utils.utils import text_to_token, generate_text, token_to_text


GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": True
}

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpt = create_pretrained_gpt2_model(device, GPT_CONFIG_124M)
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text(
        model=gpt,
        idx=text_to_token("How many malls in maryland?", tokenizer).to(device),
        max_new_tokens=100,
        context_size=GPT_CONFIG_124M["context_length"],
        top_k=50,
        temperature=1
    )
    print("Output text:\n", token_to_text(token_ids, tokenizer))


if __name__ == "__main__":
    main()






