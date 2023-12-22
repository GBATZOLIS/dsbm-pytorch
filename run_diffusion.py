import torch
import hydra
import os

from bridge.trainer_diffusion import DiffusionModelTrainer
from bridge.runners.config_getters import get_datasets, get_valid_test_datasets
from accelerate import Accelerator

def run(args):
    accelerator = Accelerator(cpu=args.device == 'cpu', split_batches=True)
    accelerator.print('Directory: ' + os.getcwd())

    init_ds, final_ds, mean_final, var_final = get_datasets(args, device=accelerator.device)
    valid_ds, test_ds = get_valid_test_datasets(args, device=accelerator.device)

    diffusion_trainer = DiffusionModelTrainer(init_ds, final_ds, mean_final, var_final, args, accelerator=accelerator,
                        valid_ds=valid_ds, test_ds=test_ds)
    
    accelerator.print(accelerator.state)
    #accelerator.print(ipf.net['b'])
    accelerator.print('Number of parameters:', sum(p.numel() for p in diffusion_trainer.model.parameters() if p.requires_grad))
    
    if args.phase == 'train':
        diffusion_trainer.train()
    elif args.phase == 'test':
        diffusion_trainer.test()
    


