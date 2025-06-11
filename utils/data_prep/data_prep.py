import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken


# Using tiktoken over custom tokenizer because I need BPE tokenizer to
# account for unknown words, which would be hard/tedious for me to make
def tokenize(text):
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    return token_ids

class AaronGPTDataset(Dataset):
    def __init__(self, text, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenize(text)

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
    dataset = AaronGPTDataset(text, max_length, stride)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)
    return dataloader



