import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
import re

class AaronGPTDataset(Dataset):
    def __init__(self, text, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode(text, allowed_special={'<|endoftext|>'})

        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, index):
        return self.input_ids[index], self.target_ids[index]
    

def create_dataloader(text, batch_size=32, max_length=64, stride=64, shuffle=True, drop_last=True, num_workers=0):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset = AaronGPTDataset(text, tokenizer, max_length, stride)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        drop_last=drop_last, 
        num_workers=num_workers
    )
    return dataloader


def format_data(file):
    final_text_concat = ""
    title_pattern = r"^\s*=.*=\s*$"
    eos_flag = False

    for line in file:
        stripped = line.strip()
        stripped = re.sub(r"@-@", "-", stripped)
        stripped = re.sub(r"@\.\s*@", ".", stripped)
        stripped = re.sub(r"@,\s*@", ",", stripped)
        stripped = re.sub(r'" *([^"]*?) *"', r'"\1"', stripped)
        stripped = re.sub(r"\s+(?=[,\.\]\)\?\$\#\%\'\:\!\;\)])", "", stripped)
        stripped = re.sub(r"(?<!-)\s*-\s*(?!-)", "-", stripped)
        stripped = re.sub(r"(?<=[\[\(])\s+", "", stripped)
        stripped = re.sub(r"(\d)\.\s+(\d)", r"\1.\2", stripped)
        stripped = re.sub(r"(\d)\,\s+(\d)", r"\1.\2", stripped)

        if len(stripped) > 0:
            if re.match(title_pattern, stripped):
                if not eos_flag:
                    final_text_concat += "<|endoftext|> "
                eos_flag = True
            else:
                final_text_concat += stripped + " "
                eos_flag = False

    final_text_concat += "<|endoftext|> "
    return final_text_concat[8:]
