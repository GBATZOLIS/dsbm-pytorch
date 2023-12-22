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
from .runners.pl_EMA import pl_EMA
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
import matplotlib.pyplot as plt
import optuna
from optuna.integration import PyTorchLightningPruningCallback
from pytorch_lightning.loggers import WandbLogger
import pickle
import re
from multiprocessing import Process
import copy 

class MetricsCallback(Callback):
    def __init__(self, freq=1, feature=2048, metrics='all'):
        super().__init__()
        self.freq = freq

        if metrics=='all':
            metrics = ['fid_recon', 'fid_samples', 'fid_recon_vs_samples', 'lpips', 'visualisation']
        
        self.metrics = metrics
            
        # Initialize FID metric for reconstructions
        if 'fid_recon' in metrics:
            self.fid_recon = FID(feature=feature)
            self.fid_recon.reset_real_features = True
        
        # Initialize FID metric for samples
        if 'fid_samples' in metrics:
            self.fid_samples = FID(feature=feature)
            self.fid_samples.reset_real_features = True
        
        # Initialize FID metric for reconstructions vs samples
        if 'fid_recon_vs_samples' in metrics:
            self.fid_recon_vs_samples = FID(feature=feature)
            self.fid_recon_vs_samples.reset_real_features = True

        # Initialize LPIPS metric
        if 'lpips' in metrics:
            self.lpips = lpips.LPIPS(net='vgg')

    def move_metrics_to_device(self, device):
        # Move metrics to the appropriate device once
        for metric in self.metrics:
            if metric != 'visualisation':
                getattr(self, metric).to(device)

    def on_validation_epoch_end(self, trainer, pl_module):
        # Skip if currently in sanity check phase
        if trainer.sanity_checking:
            return

        if (trainer.current_epoch+1) % self.freq == 0:          
            self.move_metrics_to_device(pl_module.device)

            pl_module.eval()  # Set to evaluation mode
            with torch.no_grad():
                lpips_scores = []
                for i, batch in tqdm(enumerate(trainer.train_dataloader)):
                    batch = pl_module.handle_batch(batch)
                    imgs = batch.to(pl_module.device)
                    reconstructions = pl_module.reconstruct(imgs).detach()

                    # For reconstructions
                    if 'fid_recon' in self.metrics:
                        self.fid_recon.update(to_uint8_tensor(reconstructions), to_uint8_tensor(imgs))

                    # Compute LPIPS distance
                    if 'lpips' in self.metrics:
                        lpips_score = self.lpips(reconstructions, imgs).mean().item()
                        lpips_scores.append(lpips_score)

                    # For samples
                    if 'fid_samples' in self.metrics:
                        num_samples = imgs.shape[0]  # Use the batch size from train_dataloader
                        samples = pl_module.sample(num_samples).detach()
                        self.fid_samples.update(to_uint8_tensor(samples), to_uint8_tensor(imgs))

                    # For reconstructions vs samples
                    if 'fid_recon_vs_samples' in self.metrics:
                        self.fid_recon_vs_samples.update(to_uint8_tensor(reconstructions), to_uint8_tensor(samples))

                if 'fid_recon' in self.metrics:
                    fid_score_recon = self.fid_recon.compute()
                    pl_module.logger.experiment.log({"fid_recon": fid_score_recon, "step": trainer.global_step})
                    
                
                if 'lpips' in self.metrics:
                    avg_lpips_score = sum(lpips_scores) / len(lpips_scores)
                    pl_module.logger.experiment.log({"lpips": avg_lpips_score, "step": trainer.global_step})
                    
                if 'fid_samples' in self.metrics:
                    fid_score_samples = self.fid_samples.compute()
                    pl_module.logger.experiment.log({"fid_samples": fid_score_samples, "step": trainer.global_step})
                    trainer.callback_metrics["fid_samples"] = fid_score_samples
                
                if 'fid_recon_vs_samples' in self.metrics:
                    fid_score_recon_vs_samples = self.fid_recon_vs_samples.compute()
                    pl_module.logger.experiment.log({"fid_recon_vs_samples": fid_score_recon_vs_samples, "step": trainer.global_step})
                    trainer.callback_metrics["fid_recon_vs_samples"] = fid_score_recon_vs_samples

                # Visualization
                if 'visualisation' in self.metrics:
                    dataloader = trainer.train_dataloader
                    batch = next(iter(dataloader))
                    batch = pl_module.handle_batch(batch)
                    grid_batch = torchvision.utils.make_grid(batch, nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
                    pl_module.logger.experiment.log({"step": trainer.global_step, "original": [wandb.Image(grid_batch)]})
                    
                    batch = batch.to(pl_module.device)
                    reconstruction = pl_module.reconstruct(batch)
                    sample = reconstruction.cpu()
                    grid_batch = torchvision.utils.make_grid(sample, nrow=int(np.sqrt(sample.size(0))), normalize=True, scale_each=True)
                    pl_module.logger.experiment.log({"step": trainer.global_step, "reconstruction": [wandb.Image(grid_batch)]})

                    # Visualize the latent space if the decoder has 'z_shape'
                    if hasattr(pl_module.decoder, 'z_shape'):
                        mean_z, log_var_z = pl_module.encode(batch)
                        z = torch.randn_like(mean_z, device=pl_module.device) * torch.sqrt(log_var_z.exp()) + mean_z
                        z = z.cpu()

                        num_channels = z.size(1)

                        if num_channels < 3:
                            # Create a grid for the first channel
                            grid = torchvision.utils.make_grid(z[:, 0:1, :, :], nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
                        else:
                            # Create a grid for the first three channels as an RGB image
                            grid = torchvision.utils.make_grid(z[:, 0:3, :, :], nrow=int(np.sqrt(batch.size(0))), normalize=True, scale_each=True)
                        
                        pl_module.logger.experiment.log({"step": trainer.global_step, "latent_space": [wandb.Image(grid)]})


            pl_module.train()  # Set back to training mode
    
def prepare_data_loaders(args):
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
    test_dataloader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return {'train':train_dataloader,
            'val':val_dataloader,
            'test':test_dataloader}

class SimpleEMA(pl.Callback):
    def __init__(self, decay):
        super().__init__()
        self.decay = decay
        self.ema_weights = None

    def on_train_start(self, trainer, pl_module):
        # Initialize EMA weights with the model's weights at training start
        self.ema_weights = copy.deepcopy(pl_module.state_dict())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, unused=0):
        # Update EMA weights after each training batch
        with torch.no_grad():
            model_state_dict = pl_module.state_dict()
            for name, param in model_state_dict.items():
                if name in self.ema_weights:
                    self.ema_weights[name].mul_(self.decay).add_(param, alpha=1 - self.decay)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        # Replace the model state dict in the checkpoint with EMA weights
        checkpoint['state_dict'] = self.ema_weights

def train_vae(args):
    vae_model = get_vae_model(args)

    dataloaders = prepare_data_loaders(args)
    #print(dataloaders)
    train_dataloader = dataloaders['train']
    val_dataloader = dataloaders['val']

    args.run_name = f'kl_{args.kl_weight}'
    print(f'KL weight: {args.kl_weight}')
    print(f'Latent dimension: {args.latent_dim}')
    
    logger = get_logger(args, 'logs')
    checkpoint_callback = get_checkpoint_callback(logger)
    early_stop_callback = EarlyStopping(monitor='val_loss',
                                        patience=args.early_stopping_patience)
    lr_monitor = LearningRateMonitor(logging_interval='step')

    metrics = ['lpips', 'visualisation']
    fid_callback = MetricsCallback(freq=1, metrics=metrics)

    trainer = pl.Trainer(accelerator='gpu', devices=1,
                        max_epochs=100000,
                        logger=logger,
                        #resume_from_checkpoint=args.checkpoint,
                        accumulate_grad_batches=args.accumulate_grad_batches,
                        callbacks=[checkpoint_callback,
                                   lr_monitor,
                                   early_stop_callback,
                                   fid_callback,
                                   pl_EMA(decay=0.999)
                                   ]
                        )
    
    trainer.fit(vae_model, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

class VAEHyperparameterTuner:
    def __init__(self, args):
        args.wandb_id = None
        self.args = args
        
        # Initialize the study object only if we're not running in parallel
        if not hasattr(self.args, 'tuning_gpus') or len(self.args.tuning_gpus) <= 1:
            self.study = optuna.create_study(direction='minimize')
        
        self.dataloaders = prepare_data_loaders(args)
        self.default_project_name = args.Dataset + 'VAE_tuning_study'  # Note: Python variables should be lowercase

        #args.run_name = 'summary'
        #self.logger = get_logger(args, 'logs', project_name=self.default_project_name)
        #self.main_logger_dir = self.logger.experiment.dir
        #studies_directory = os.path.join(self.main_logger_dir, 'studies')
        #self.studies_directory = studies_directory
        #os.makedirs(studies_directory, exist_ok=True)
        #wandb.finish()

        self.kl_weight_min = args.kl_weight_min
        self.kl_weight_max = args.kl_weight_max
        self.fid_feature = args.tuning_fid_feature
        self.freq = args.vae_tuning_freq
        self.max_epochs = self.freq + 1
        self.n_trials = args.n_trials
        
        # Determine the number of GPUs available for tuning
        if hasattr(self.args, 'tuning_gpus'):
            self.tuning_gpus = self.args.tuning_gpus
        else:
            self.tuning_gpus = [0] 

    def get_vae_model(self, kl_weight):
        # Your model initialization logic goes here, e.g.:
        self.args.kl_weight = kl_weight
        return get_vae_model(self.args)
    
    def get_optuna_report_callback(self,):
        class OptunaReportCallback(Callback):
            def __init__(self, trial, metric_names, weights=None, freq=1):
                """
                Args:
                    trial (optuna.trial.Trial): The Optuna trial object.
                    metric_names (list of str): Names of the metrics to report.
                    weights (list of float, optional): Weights for each metric to calculate the weighted sum.
                """
                super().__init__()
                self.trial = trial
                self.freq = freq
                self.metric_names = metric_names
                self.weights = weights if weights is not None else [1.0 for _ in metric_names]

                if len(self.weights) != len(self.metric_names):
                    raise ValueError("The number of weights must match the number of metric names")

            def on_validation_epoch_end(self, trainer, pl_module):
                # Calculate the weighted sum of the metrics
                if (trainer.current_epoch + 1) % self.freq == 0:
                    weighted_sum = sum(
                        self.weights[i] * trainer.callback_metrics.get(metric_name, 0)
                        for i, metric_name in enumerate(self.metric_names)
                    )

                    for metric_name in self.metric_names:
                        metric_value = trainer.callback_metrics.get(metric_name, 0)
                        #print(f"{metric_name}: {metric_value} (type: {type(metric_value)})")

                    # Report the weighted sum to the Optuna trial
                    self.trial.report(weighted_sum, trainer.current_epoch)

                    # Additionally, save the weighted sum in callback_metrics so the pruning callback can access it
                    trainer.callback_metrics['weighted_fid'] = weighted_sum

                    # Check if the trial should be pruned
                    if self.trial.should_prune():
                        raise optuna.exceptions.TrialPruned()
        
        return OptunaReportCallback
    
    def get_callbacks(self, trial):
        monitor_metrics = ['fid_samples', 'fid_recon_vs_samples']

        # Initialize the MetricsCallback
        metrics_callback = MetricsCallback(freq=self.freq, feature=self.fid_feature, metrics=monitor_metrics)
        
        # Initialize the PyTorchLightningPruningCallback
        pruning_callback = PyTorchLightningPruningCallback(trial, monitor='weighted_fid')
        
        # Initialize the OptunaReportCallback
        report_callback = self.get_optuna_report_callback()(
            trial=trial,
            metric_names=['fid_samples', 'fid_recon_vs_samples'],
            weights=[0.5, 0.5],  # Assuming you want equal weight for each metric
            freq=self.freq
        )

        # The order of callbacks here is important. Make sure that metrics_callback is before report_callback
        return [metrics_callback, report_callback, pruning_callback]


    def objective(self, trial):
        kl_weight = trial.suggest_float('kl_weight', self.kl_weight_min, self.kl_weight_max)
        model = self.get_vae_model(kl_weight)

        self.args.run_name = 'kl_%.5f' % kl_weight
        trial_logger = get_logger(self.args, 'logs', self.default_project_name)

        callbacks = self.get_callbacks(trial)

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            logger=trial_logger,
        )

        # Fit the model
        trainer.fit(
            model,
            train_dataloaders=self.dataloaders['train'],
            val_dataloaders=self.dataloaders['val']
        )

        # Extract the weighted FID score from the callback_metrics
        # This is the score that OptunaReportCallback has calculated and reported to the trial
        weighted_fid_score = trainer.callback_metrics.get('weighted_fid')

        # Make sure the weighted FID score is found
        if weighted_fid_score is None:
            raise ValueError("Weighted FID score not found; make sure the OptunaReportCallback is being used properly.")

        wandb.finish()

        # The objective value is the weighted FID score
        return weighted_fid_score

    def plot_and_save_optuna_results(self, study, plot_filename):
        scores = [trial.value for trial in study.trials]
        kl_weights = [trial.params['kl_weight'] for trial in study.trials]

        plt.figure(figsize=(10, 5))
        plt.scatter(kl_weights, scores, color='blue')
        plt.title('Objective Value vs KL Weight')
        plt.xlabel('KL Weight')
        plt.ylabel('Objective Value (FID Score)')
        plt.grid(True)

        plot_path = os.path.join(self.main_logger_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()

        #self.logger.experiment.log({'optuna_study_plot': wandb.Image(plot_path)})

    def get_best_kl_weight(self):
        return self.best_trial.params['kl_weight']

    def tune(self, n_trials):
        self.study.optimize(self.objective, n_trials=n_trials)

    def single_run(self, n_trials):
        self.tune(n_trials=n_trials)
        #plot_filename='optuna_kl_range_%.4f_%.4f.png' % (self.args.kl_weight_min, self.args.kl_weight_max)
        #self.plot_and_save_optuna_results(self.study, plot_filename)
        best_trial = self.study.best_trial
        self.best_trial = best_trial

        #report results
        print(f'Best trial kl_weight: {best_trial.params["kl_weight"]}')
        print(f'Best FID score: {best_trial.value}')
    
    def run(self,):
        if len(self.tuning_gpus) > 1:
            # If multiple GPUs are specified, run in parallel
            self.parallel_run(gpus=self.tuning_gpus, n_trials=self.args.n_trials, plot_filename='optuna_results.png')
        else:
            # If a single GPU is specified, run the single_run method
            self.single_run(n_trials=self.args.n_trials)
    
    def parallel_objective(self, trial, gpu_index, interval):
        # Set CUDA_VISIBLE_DEVICES environment variable to the specific GPU
        #os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        kl_weight = trial.suggest_float('kl_weight', interval[0], interval[1], log=True)
        model = self.get_vae_model(kl_weight)

        self.args.run_name = 'kl_%.3f' % kl_weight
        trial_logger = get_logger(self.args, 'logs', self.default_project_name)

        callbacks = self.get_callbacks(trial)

        trainer = pl.Trainer(
            accelerator='gpu',
            devices=1,  # Assuming you want to use only 1 GPU
            gpus=[gpu_index],  # Specify the GPU index here
            max_epochs=self.max_epochs,
            callbacks=callbacks,
            logger=trial_logger,
            # You might also need to set 'strategy' if you have a specific distributed training approach
        )
        
        # Fit the model
        trainer.fit(
            model,
            train_dataloaders=self.dataloaders['train'],
            val_dataloaders=self.dataloaders['val']
        )

        # Extract the weighted FID score from the callback_metrics
        # This is the score that OptunaReportCallback has calculated and reported to the trial
        weighted_fid_score = trainer.callback_metrics.get('weighted_fid')

        # Make sure the weighted FID score is found
        if weighted_fid_score is None:
            raise ValueError("Weighted FID score not found; make sure the OptunaReportCallback is being used properly.")

        wandb.finish()

        # The objective value is the weighted FID score
        return weighted_fid_score

    def parallel_tune_with_interval(self, interval, gpu_index, n_trials):
        # Create a separate Optuna study for each GPU
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

        study = optuna.create_study(direction='minimize')
        
        # Objective wrapper to pass the gpu_index and interval
        objective_wrapper = lambda trial: self.parallel_objective(trial, gpu_index, interval)
        
        study.optimize(objective_wrapper, n_trials=n_trials)
        
        # Save the study or return results that can be aggregated later
        self.save_study_results(study, gpu_index)

    def parallel_run(self, gpus, n_trials=100, plot_filename='optuna_results.png'):
        # Determine the log-scale intervals
        d=len(gpus)
        log_intervals = np.logspace(np.log10(self.kl_weight_min), np.log10(self.kl_weight_max), d + 1)

        # Define the number of trials per GPU
        n_trials_per_gpu = n_trials // d

        # Create a process for each GPU
        processes = []
        for i, gpu_index in enumerate(gpus):
            interval = (log_intervals[i], log_intervals[i + 1])
            print(f'gpu: {gpu_index}')
            p = Process(target=self.parallel_tune_with_interval, args=(interval, gpu_index, n_trials_per_gpu))
            processes.append(p)
            try:
                p.start()
            except Exception as e:
                print(f"An error occurred while starting the process: {e}")


        # Wait for all processes to finish
        for p in processes:
            p.join()

        studies_directory = self.studies_directory
        study_list = self.load_studies_from_directory(studies_directory)
        
        # Plot and save the overall results
        self.plot_and_save_parallel_optuna_results(study_list, plot_filename)

        # Collect and aggregate the results from the separate studies
        best_trial = self.parallel_get_best_trial(study_list)
        self.best_trial = best_trial

        # Report the best results
        print(f'Best trial kl_weight: {best_trial.params["kl_weight"]}')
        print(f'Best FID score: {best_trial.value}')

    def parallel_get_best_trial(self, study_list):
        # Find the best trial across all studies
        best_trial = min(
            (trial for study in study_list for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE),
            key=lambda trial: trial.value,
            default=None
        )

        if best_trial is None:
            raise ValueError("No completed trials found across all studies")

        return best_trial

    def plot_and_save_parallel_optuna_results(self, study_list, plot_filename):
        all_scores = []
        all_kl_weights = []

        # Extract scores and kl_weights from all studies
        for study in study_list:
            all_scores.extend([trial.value for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])
            all_kl_weights.extend([trial.params['kl_weight'] for trial in study.trials if trial.state == optuna.trial.TrialState.COMPLETE])

        plt.figure(figsize=(10, 5))
        plt.scatter(all_kl_weights, all_scores, color='blue')
        plt.title(f'Objective Value vs KL Weight')
        plt.xlabel('KL Weight')
        plt.ylabel('Objective Value (FID Score)')
        plt.grid(True)

        # Ensure the directory exists
        os.makedirs(self.main_logger_dir, exist_ok=True)
        plot_path = os.path.join(self.main_logger_dir, plot_filename)
        plt.savefig(plot_path)
        plt.close()
    
    def save_study_results(self, study, gpu_index):
        # Define the directory path for saving the study relative to the wandb run directory
        studies_dir = self.studies_directory

        # Make sure the 'studies' directory exists
        os.makedirs(studies_dir, exist_ok=True)

        # Define the filename for saving the study
        study_filename = f"study_gpu_{gpu_index}.pkl"
        study_filepath = os.path.join(studies_dir, study_filename)

        # Save the study to a file
        with open(study_filepath, 'wb') as f:
            pickle.dump(study, f)
    
    def load_studies_from_directory(self, directory):
        """
        Loads all study objects from a given directory that match the filename pattern 'study_gpu_{x}.pkl'.
        :param directory: The directory from which to load the study objects.
        :return: A list of study objects.
        """
        studies = []
        study_file_pattern = re.compile(r'^study_gpu_\d+\.pkl$')

        for filename in os.listdir(directory):
            if study_file_pattern.match(filename):
                study_filepath = os.path.join(directory, filename)
                with open(study_filepath, 'rb') as f:
                    study = pickle.load(f)
                    studies.append(study)
                print(f"Loaded study from {study_filepath}")

        return studies