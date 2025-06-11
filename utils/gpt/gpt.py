import torch
from utils.config import GPT_CONFIG as config
from utils.gpt.transformer_block.transformer_block import TransformerBlock
from utils.gpt.transformer_block.layer_norm import LayerNorm

class AaronGPTModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = torch.nn.Embedding(config["vocab_size"], config["emb_dim"])
        self.pos_embed = torch.nn.Embedding(config["context_length"], config["emb_dim"])
        self.drop_emb = torch.nn.Dropout(config["drop_rate"])

        self.transformer_block = torch.nn.Sequential(
            * [TransformerBlock() for _ in range(config["n_layers"])]
        )
        self.final_layer_norm = LayerNorm(config["emb_dim"])
        self.out_head = torch.nn.Linear(
            config["emb_dim"],
            config["vocab_size"],
            bias=False
        )

    def forward(self, in_idx):
        batch, seq_len = in_idx.shape
        token_embed = self.token_embed(in_idx)
        pos_embed = self.pos_embed(torch.arange(seq_len, device=in_idx.device))
        embedded_tokens = token_embed + pos_embed
        embedded_tokens = self.drop_emb(embedded_tokens)
        embedded_tokens = self.transformer_block(embedded_tokens)
        embedded_tokens = self.final_layer_norm(embedded_tokens)
        logits = self.out_head(embedded_tokens)
        return logits




