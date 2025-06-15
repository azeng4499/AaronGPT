import torch
import tiktoken
import os
import sys
from main import GPT_CONFIG_124M

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from gpt.gpt import AaronGPTModel
from utils.utils import generate_text, token_to_text

def model_test(
    input_text,
    model_path,
    max_new_tokens=100,
    temperature=1.0,
    top_k=40,
    context_size=1024 
):
    if not input_text or not isinstance(input_text, str):
        raise ValueError("You must provide a valid input text.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = tiktoken.get_encoding("gpt2")

    input_ids = tokenizer.encode(input_text, allowed_special={'<|EOS|>'})
    input_ids = torch.tensor([input_ids], dtype=torch.long).to(device)

    model = AaronGPTModel(GPT_CONFIG_124M)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with torch.no_grad():
        output_ids = generate_text(
            model=model,
            idx=input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            context_size=context_size
        )

    output_text = token_to_text(output_ids, tokenizer)
    print("=== Generated Text ===\n", output_text)
    return output_text

if __name__ == "__main__":
    model_test(
        input_text="The titanic was", 
        model_path="final_trained_model.pt"
    )