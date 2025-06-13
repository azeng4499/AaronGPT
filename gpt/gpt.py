import torch
from config.config import GPT_CONFIG as config
from gpt.transformer_block.transformer_block import TransformerBlock
from gpt.transformer_block.layer_norm import LayerNorm

class AaronGPTModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embed = torch.nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed = torch.nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = torch.nn.Dropout(config["drop_rate"])

        self.transformer_blocks = torch.nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["n_layers"])])
        
        self.final_layer_norm = LayerNorm(config["emb_dim"])
        self.out_head = torch.nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False
        )

    def forward(self, in_idx):
        batch, seq_len = in_idx.shape
        token_embed = self.token_embed(in_idx)
        pos_embed = self.pos_embed(
            torch.arange(seq_len, device=in_idx.device)
        )
        x = token_embed + pos_embed
        x = self.drop_emb(x)
        x = self.transformer_blocks(x)
        x = self.final_layer_norm(x)
        logits = self.out_head(x)
        return logits




