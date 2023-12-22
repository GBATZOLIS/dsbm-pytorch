import torch
import hydra
import os
from bridge.trainer_vae import train_vae, VAEHyperparameterTuner


def run(args):
    #tuner = VAEHyperparameterTuner(args)
    #tuner.run() #finds the best kl_weight
    #kl_weight = tuner.get_best_kl_weight()
    #args.kl_weight = kl_weight #use the optimal kl_weight for the training of the VAE model
    train_vae(args)
