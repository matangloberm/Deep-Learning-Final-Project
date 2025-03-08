# -*- coding: utf-8 -*-
"""embedding_model.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/137_-lX0m9mUCgMBrWigQO2gyn-wifogz
"""

import torch
import config2 as config
import importlib
import sys


def upload_model():
    """Load the selected DINOv2 model dynamically from Torch Hub."""
    model_map = {
        "vits14": "dinov2_vits14",
        "vitb14": "dinov2_vitb14",
        "vitl14": "dinov2_vitl14",
    }

    if config.args.embedding_model not in model_map:
        raise ValueError(f"Unsupported model type: {config.args.embedding_model}. Choose from vits14, vitb14, vitl14.")

    # Load the correct model from Torch Hub
    model_name = model_map[config.args.embedding_model]
    model = torch.hub.load("facebookresearch/dinov2", model_name).to(config.args.device)
    model.eval()  # Set model to evaluation mode

    print(f"Loaded {config.args.embedding_model} model from Torch Hub")
    return model