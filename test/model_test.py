import sys
import os
import torch
import tiktoken
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from gpt.gpt import AaronGPTModel
from utils.utils import generate_text, token_to_text

def model_test(
    input_text,
    model_path="trained_model.pt",
    max_new_tokens=50,
    temperature=1.0,
    top_k=40,
    context_size=1024
):
    if not input_text or not isinstance(input_text, str):
        raise ValueError("You must provide an input text.")

    tokenizer = tiktoken.get_encoding("gpt2")
    input_ids = tokenizer.encode(input_text, allowed_special={'<|EOS|>'})
    input_ids = torch.tensor([input_ids])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AaronGPTModel()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    output_ids = generate_text(
        model=model,
        idx=input_ids.to(device),
        max_new_tokens=max_new_tokens,
        context_size=context_size,
        temperature=temperature,
        top_k=top_k,
    )

    output_text = token_to_text(output_ids, tokenizer)
    print("Generated text:\n", output_text)
    return output_ids


model_test("Hello")