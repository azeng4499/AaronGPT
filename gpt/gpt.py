# This implementation is adapted from the GPT model in
# "Build a Large Language Model from Scratch" by Sebastian Raschka.
# Some parts have been modified or extended to better align with the design and 
# functionality requirements of the Better Threads Project.

import torch
from gpt.transformer_block.transformer_block import TransformerBlock
from gpt.transformer_block.layer_norm import LayerNorm

class AaronGPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = torch.nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed = torch.nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = torch.nn.Dropout(config["drop_rate"])
        self.num_classes = config.get("num_classes", None)

        self.transformer_blocks = torch.nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])])
        
        self.final_layer_norm = LayerNorm(config["emb_dim"])

        if self.num_classes is not None:
            self.out_head = torch.nn.Linear(config["emb_dim"], self.num_classes, bias=True)
        else:
            self.out_head = torch.nn.Linear(config["emb_dim"], config["vocab_size"], bias=False)

    def forward(self, in_idx, return_hidden=False):
        batch, seq_len = in_idx.shape
        token_embed = self.token_embed(in_idx)
        pos_embed = self.pos_embed(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = token_embed + pos_embed
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)

        if return_hidden:
            return x
        
        logits = self.out_head(x)
        return logits
    
    def get_hidden_states(self, in_idx):
        return self.forward(in_idx, return_hidden=True)




