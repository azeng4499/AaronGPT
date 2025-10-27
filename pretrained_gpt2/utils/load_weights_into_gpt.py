# This implementation is adapted from the GPT model in
# "Build a Large Language Model from Scratch" by Sebastian Raschka.
# Some parts have been modified or extended to better align with the design and 
# functionality requirements of the Better Threads Project.

import torch
import numpy as np
from pretrained_gpt2.utils.weights_downloader import download_and_load_gpt2
from gpt.gpt import AaronGPTModel

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.pos_embed.weight = assign(gpt.pos_embed.weight, params['wpe'])
    gpt.token_embed.weight = assign(gpt.token_embed.weight, params['wte'])
    for b in range(len(params["blocks"])):
        q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        gpt.transformer_blocks[b].multi_head_attention.W_q.weight = assign(gpt.transformer_blocks[b].multi_head_attention.W_q.weight, q_w.T)
        gpt.transformer_blocks[b].multi_head_attention.W_k.weight = assign(gpt.transformer_blocks[b].multi_head_attention.W_k.weight, k_w.T)
        gpt.transformer_blocks[b].multi_head_attention.W_v.weight = assign(gpt.transformer_blocks[b].multi_head_attention.W_v.weight, v_w.T)
        q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        gpt.transformer_blocks[b].multi_head_attention.W_q.bias = assign(gpt.transformer_blocks[b].multi_head_attention.W_q.bias, q_b)
        gpt.transformer_blocks[b].multi_head_attention.W_k.bias = assign(gpt.transformer_blocks[b].multi_head_attention.W_k.bias, k_b)
        gpt.transformer_blocks[b].multi_head_attention.W_v.bias = assign(gpt.transformer_blocks[b].multi_head_attention.W_v.bias, v_b)
        gpt.transformer_blocks[b].multi_head_attention.out_proj.weight = assign(
            gpt.transformer_blocks[b].multi_head_attention.out_proj.weight, 
            params["blocks"][b]["attn"]["c_proj"]["w"].T
        )
        gpt.transformer_blocks[b].multi_head_attention.out_proj.bias = assign(
            gpt.transformer_blocks[b].multi_head_attention.out_proj.bias, 
            params["blocks"][b]["attn"]["c_proj"]["b"]
        )
        gpt.transformer_blocks[b].feed_forward.layers[0].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].weight, 
            params["blocks"][b]["mlp"]["c_fc"]["w"].T
        )
        gpt.transformer_blocks[b].feed_forward.layers[0].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[0].bias, 
            params["blocks"][b]["mlp"]["c_fc"]["b"]
        )
        gpt.transformer_blocks[b].feed_forward.layers[2].weight = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].weight, 
            params["blocks"][b]["mlp"]["c_proj"]["w"].T
        )
        gpt.transformer_blocks[b].feed_forward.layers[2].bias = assign(
            gpt.transformer_blocks[b].feed_forward.layers[2].bias, 
            params["blocks"][b]["mlp"]["c_proj"]["b"]
        )
        gpt.transformer_blocks[b].layer_norm_1.scale = assign(
            gpt.transformer_blocks[b].layer_norm_1.scale, 
            params["blocks"][b]["ln_1"]["g"]
        )
        gpt.transformer_blocks[b].layer_norm_1.shift = assign(
            gpt.transformer_blocks[b].layer_norm_1.shift, 
            params["blocks"][b]["ln_1"]["b"]
        )
        gpt.transformer_blocks[b].layer_norm_2.scale = assign(
            gpt.transformer_blocks[b].layer_norm_2.scale, 
            params["blocks"][b]["ln_2"]["g"]
        )
        gpt.transformer_blocks[b].layer_norm_2.shift = assign(
            gpt.transformer_blocks[b].layer_norm_2.shift, 
            params["blocks"][b]["ln_2"]["b"]
        )

    gpt.final_layer_norm.scale = assign(gpt.final_layer_norm.scale, params["g"])
    gpt.final_layer_norm.shift = assign(gpt.final_layer_norm.shift, params["b"])
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

def create_pretrained_gpt2_model(device, config):
    settings, params = download_and_load_gpt2(model_size="124M", models_dir="pretrained_gpt2")
    gpt = AaronGPTModel(config)
    gpt.eval()
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    print("Finished loading weights.")
    return gpt