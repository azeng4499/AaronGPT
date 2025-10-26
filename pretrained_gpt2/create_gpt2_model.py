import pickle
import torch
import os
from pretrained_gpt2.utils.load_weights_into_gpt import download_and_load_gpt2, load_weights_into_gpt
from gpt.gpt import AaronGPTModel

def create_gpt2_model(config):
    pickle_path_settings = "/root/AaronGPT/pickle_vars/pretrained_gpt2_settings.pkl"
    pickle_path_params = "/root/AaronGPT/pickle_vars/pretrained_gpt2_params.pkl"
    pickle_path_model = "/root/AaronGPT/pickle_vars/pretrained_gpt2_mdoel.pkl"

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
        model = AaronGPTModel(config)
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