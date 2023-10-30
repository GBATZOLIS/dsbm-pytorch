import os, sys, warnings, time
import re
from collections import OrderedDict
from functools import partial

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from .data.utils import to_uint8_tensor
from .runners import *
from .runners.config_getters import get_vae_model, get_logger, get_checkpoint_callback, get_datasets, get_valid_test_datasets

#new imports 
from torch.utils.data import random_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor

import torchvision
import numpy as np
import torch
from pytorch_lightning import Callback
from .data.metrics import FID
import lpips
from tqdm import tqdm 
import wandb

class MetricsCallback(Callback):
    def __init__(self, freq=1):
        super().__init__()
        self.freq = freq
        
        # Initialize FID metric for reconstructions
        self.fid_recon = FID()
        self.fid_recon.reset_real_features=True
        
        # Initialize FID metric for samples
        self.fid_samples = FID()
        self.fid_samples.reset_real_features=True

        # Initialize LPIPS metric
        self.lpips_distance_fn = lpips.LPIPS(net='vgg')

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch+1) % self.freq == 0:
            self.lpips_distance_fn.to(pl_module.device)
            self.fid_recon.to(pl_module.device)
            self.fid_samples.to(pl_module.device)

            pl_module.eval()  # Set to evaluation mode
            with torch.no_grad():
                #CODE for FID calculation

                # For reconstructions
                real_imgs = []
                recon_imgs = []
                lpips_scores = []

                # For samples
                samples_imgs = []

                for i, batch in tqdm(enumerate(trainer.train_dataloader)):
                    batch = pl_module.handle_batch(batch)
                    imgs = batch.to(pl_module.device)
                    real_imgs.append(imgs)
                    
                    # For reconstructions
                    reconstructions = pl_module.reconstruct(imgs).detach()
                    self.fid_recon.update(to_uint8_tensor(reconstructions), to_uint8_tensor(imgs))
                    recon_imgs.append(reconstructions)

                    # Compute LPIPS distance
                    lpips_score = self.lpips_distance_fn(reconstructions, imgs).mean().item()
                    lpips_scores.append(lpips_score)

                    # For samples
                    num_samples = imgs.shape[0]  # Use the batch size from train_dataloader
                    samples = pl_module.sample(num_samples).detach()
                    self.fid_samples.update(to_uint8_tensor(samples), to_uint8_tensor(imgs))
                    samples_imgs.append(samples)

                real_imgs = torch.cat(real_imgs, axis=0)
                recon_imgs = torch.cat(recon_imgs, axis=0)
                samples_imgs = torch.cat(samples_imgs, axis=0)

                fid_score_recon = self.fid_recon.compute()
                fid_score_samples = self.fid_samples.compute()

                pl_module.logger.experiment.log({"FID_Reconstructions": fid_score_recon, "step": trainer.global_step})
                pl_module.logger.experiment.log({"FID_Samples": fid_score_samples, "step": trainer.global_step})
                
                
                # Log average LPIPS score
                avg_lpips_score = sum(lpips_scores) / len(lpips_scores)
                pl_module.logger.experiment.log({"LPIPS_Reconstructions": avg_lpips_score, "step": trainer.global_step})


                #CODE for VISUALISATION 
                #visualisation of an image from the validation set and its reconstruction
                dataloader = trainer.train_dataloader
                batch = next(iter(dataloader))
                batch = pl_module.handle_batch(batch)
                grid_batch = torchvision.utils.make_grid(batch, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
                pl_module.logger.experiment.log({"original": [wandb.Image(grid_batch)]})
                batch = batch.to(pl_module.device)
                reconstruction = pl_module.reconstruct(batch)
                sample = reconstruction.cpu()
                grid_batch = torchvision.utils.make_grid(sample, nrow=int(np.sqrt(sample.size(0))), normalize=True, scale_each=True)
                pl_module.logger.experiment.log({"reconstruction": [wandb.Image(grid_batch)]})

            pl_module.train()  # Set back to training mode
            

def train_vae(args):
    vae_model = get_vae_model(args)

    train_ds, _, _, _ = get_datasets(args)
    valid_ds, test_ds = get_valid_test_datasets(args)

    if valid_ds is None:
        generator = torch.Generator().manual_seed(42)
        dataset_size = len(train_ds)
        train_size = int(0.9 * dataset_size)
        valid_size = dataset_size - train_size
        train_ds, valid_ds = random_split(train_ds, [train_size, valid_size], generator=generator)

    train_dataloader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_dataloader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    #test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    logger = get_logger(args, 'logs', run_name='vae')
    checkpoint_callback = get_checkpoint_callback(logger)
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=30)
    lr_monitor = LearningRateMonitor()
    fid_callback = MetricsCallback(freq=50)

    trainer = pl.Trainer(accelerator='gpu', devices=1,
                        max_epochs=100000,
                        logger=logger,
                        #resume_from_checkpoint=args.checkpoint,
                        callbacks=[checkpoint_callback,
                                   lr_monitor,
                                   early_stop_callback,
                                   fid_callback
                                   ]
                        )
    
    trainer.fit(vae_model, train_dataloader=train_dataloader, val_dataloader=val_dataloader, ckpt_path=args.vae_checkpoint_path)

