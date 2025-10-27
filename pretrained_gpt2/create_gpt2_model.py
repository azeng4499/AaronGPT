# This implementation is adapted from the GPT model in
# "Build a Large Language Model from Scratch" by Sebastian Raschka.
# Some parts have been modified or extended to better align with the design and 
# functionality requirements of the Better Threads Project.

import pickle
import torch
import os
from pretrained_gpt2.utils.load_weights_into_gpt import download_and_load_gpt2, load_weights_into_gpt
from gpt.gpt import BTPModel

def create_gpt2_model(config):
    pickle_path_settings = "./pickle_vars/pretrained_gpt2_settings.pkl"
    pickle_path_params = "./pickle_vars/pretrained_gpt2_params.pkl"
    pickle_path_model = "./pickle_vars/pretrained_gpt2_model.pkl"

    if os.path.exists(pickle_path_settings) and os.path.exists(pickle_path_params) and os.path.exists(pickle_path_model):
        with open(pickle_path_settings, "rb") as f:
            settings = pickle.load(f)
        with open(pickle_path_params, "rb") as f:
            params = pickle.load(f)
        with open(pickle_path_model, "rb") as f:
            model = pickle.load(f)

        return model, settings, params
    else:
        settings, params = download_and_load_gpt2(
            model_size="124M", models_dir="pretrained_gpt2"
        )
        model = BTPModel(config)
        load_weights_into_gpt(model, params)

        model.eval()

        for param in model.parameters():
            param.requires_grad = False

        model.out_head = torch.nn.Linear(
            in_features=config["emb_dim"],
            out_features=2
        )

        for param in model.transformer_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_layer_norm.parameters():
            param.requires_grad = True

        with open(pickle_path_settings, "wb") as f:
            pickle.dump(settings, f)
        with open(pickle_path_params, "wb") as f:
            pickle.dump(params, f)
        with open(pickle_path_model, "wb") as f:
            pickle.dump(model, f)

        return model, settings, params