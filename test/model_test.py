import torch
import tiktoken
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpt.gpt import AaronGPTModel
from utils.utils import generate_text, token_to_text 

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 256,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False
}


def generate_from_checkpoint(
    checkpoint_path,
    prompt="The future of AI is",
    max_new_tokens=100,
    temperature=1.0,
    top_k=40,
    context_length=1024,
    device=None
):
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    model = AaronGPTModel(GPT_CONFIG_124M)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        output_ids = generate_text(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            context_size=context_length,
        )
    output_text = token_to_text(output_ids, tokenizer)
    print("\n📝 Generated Text:\n")
    print(output_text)


if __name__ == "__main__":
    generate_from_checkpoint(
        checkpoint_path="checkpoint_epoch_3.pt",
        prompt="One day I'll acheive",
        max_new_tokens=100
    )
