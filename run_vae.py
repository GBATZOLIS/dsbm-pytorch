import torch
import hydra
import os
from bridge.trainer_vae import train_vae

# Set the CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def run(args):
    train_vae(args)
