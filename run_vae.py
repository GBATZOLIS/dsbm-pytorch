import torch
import hydra
import os
from bridge.trainer_vae import train_vae, VAEHyperparameterTuner

# Set the CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

def run(args):
    #tuner = VAEHyperparameterTuner(args)
    #tuner.run() #finds the best kl_weight
    #kl_weight = tuner.get_best_kl_weight()
    #args.kl_weight = kl_weight #use the optimal kl_weight for the training of the VAE model
    train_vae(args)
