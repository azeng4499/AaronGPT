import torch
import tiktoken
from cyberbullying_detector.cb_detect_train import cb_detect_train
from pretrained_gpt2.create_gpt2_model import create_gpt2_model
from gpt.gpt import AaronGPTModel

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.0,
    "qkv_bias": True
}

def main():
    #train
    cb_detect_train()

    #run
    # model, settings, params = create_gpt2_model(GPT_CONFIG_124M)
    # tokenizer = tiktoken.get_encoding("gpt2")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # input = "Aaron sucks so much"
    # model_state_dict = torch.load("./cyberbullying_detector.pth", map_location=device)
    # model.load_state_dict(model_state_dict)

    # print(run_detector(
    #     input, 
    #     model, 
    #     tokenizer, 
    #     device, 
    #     max_length=GPT_CONFIG_124M["context_length"]
    # ))


def run_detector(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()

    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_embed.weight.shape[1]

    input_ids = input_ids[:min(
        max_length, supported_context_length
    )]

    input_ids += [pad_token_id] * (max_length - len(input_ids))

    input_tensor = torch.tensor(
        input_ids, device=device
    ).unsqueeze(0)

    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
    
    predicted_label = torch.argmax(logits, dim=-1).item()

    return "cyberbullying" if predicted_label == 1 else "not cyberbullying"


if __name__ == "__main__":
    main()
