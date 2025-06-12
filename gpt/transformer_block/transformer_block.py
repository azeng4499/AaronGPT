from gpt.transformer_block.transformer_block import MultiHeadAttention
import torch
from config.config import GPT_CONFIG as config
from gpt.transformer_block.feed_forward import FeedForward
from gpt.transformer_block.layer_norm import LayerNorm


class TransformerBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(
            dim_in = config["emb_dim"],
            dim_out = config["emb_dim"],
            context_length = config["context_length"],
            dropout = config["drop_rate"],
            num_heads = config["n_heads"],
            qkv_bias= config["qkv_bias"]
        )
        self.feed_forward = FeedForward()
        self.layer_norm_1 = LayerNorm(config["emb_dim"])
        self.layer_norm_2 = LayerNorm(config["emb_dim"])
        self.drop_shortcut = torch.nn.Dropout(config["drop_rate"])

    def forward(self, x):
        
        shortcut = x
        x = self.layer_norm_1(x)
        x = self.multi_head_attention(x)
        x = self.drop_shortcut(x)

        x = x + shortcut

        shortcut = x
        x = self.layer_norm_2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)

        x = x + shortcut

        return x


        