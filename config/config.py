GPT_CONFIG = {
    "vocab_size": 50257,
    "context_length": 512,   # Reduced from 1024
    "emb_dim": 512,          # Reduced from 1024
    "n_heads": 8,            # Reduced from 16
    "n_layers": 8,           # Reduced from 16  
    "drop_rate": 0.3,        # Increased from 0.2
    "qkv_bias": False
}