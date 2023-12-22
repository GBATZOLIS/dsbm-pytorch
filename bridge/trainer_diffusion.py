import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import os
import shutil
from functools import partial
from scipy.interpolate import PchipInterpolator
from .sde import *
from .sde import SNRSDE, cSNRSDE, VPSDE, cVPSDE, subVPSDE, csubVPSDE, VESDE, cVESDE
from .runners import *
from .runners.config_getters import get_vae_model, get_model, get_optimizer, get_plotter, get_logger
from .runners.ema import EMAHelper
import re
from math import ceil

class DiffusionModelTrainer:
    def __init__(self, init_ds, final_ds, mean_final, var_final, args, accelerator=None, valid_ds=None, test_ds=None):
        self.accelerator = accelerator
        self.device = accelerator.device if accelerator else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.args = args
        self.cdsb = self.args.cdsb  # Conditional

        self.init_ds = init_ds
        self.final_ds = final_ds
        self.valid_ds = valid_ds
        self.test_ds = test_ds

        self.mean_final = mean_final
        self.var_final = var_final
        self.std_final = torch.sqrt(var_final) if var_final is not None else None

        # Parameters for diffusion model
        self.num_steps = args.num_steps
        self.test_num_steps = args.get("test_num_steps", self.num_steps)
        self.batch_size = args.batch_size
        self.num_repeat_data = args.get("num_repeat_data", 1)
        assert self.batch_size % self.num_repeat_data == 0
        self.first_num_iter = self.args.get("first_num_iter", 1000000)
        self.normalize_x1 = self.args.get("normalize_x1", False)
        self.grad_clipping = self.args.grad_clipping

        # Initialize loggers
        self.logger = self.get_logger('train_logs') #DONE
        self.save_logger = self.get_logger('test_logs') #DONE

        #MODELS
        #SET the checkpoints right (handles resuming) 
        self.build_checkpoint() #DONE
        self.build_model() #DONE
        self.build_ema() #DONE

        #we need this to evaluate the latent diffusion model in run-time
        if args.space == 'latent':
            self.vae_model = self.build_vae_model(args) #DONE

        self.build_optimizer() #DONE
        self.build_dataloaders() #DONE
        self.compute_normalizing_constants()

        #Set the SDE (try to make it follow the style of self.langevin)
        self.configure_sde(args)

        #plotter
        self.plotter = self.get_plotter()
        self.stride = self.args.gif_stride #evaluate every stride num iterarions
        self.stride_log = self.args.log_stride
        self.validation_stride = getattr(self.args, 'validation_stride', 10000)
        self.test_num_steps_sequence = self.args.get("test_num_steps_sequence", [100]) #we test FID with this sequence of integrations steps

    def set_seed(self, seed=0):
        torch.manual_seed(seed)
        np.random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def get_logger(self, name='logs'):
        return get_logger(self.args, name)

    def get_plotter(self, fid_feature=2048):
        return get_plotter(self, self.args, fid_feature)
    
    def build_vae_model(self, args):
        # Instantiate and load the VAE model from the checkpoint
        vae_model = get_vae_model(args)
        vae_model = vae_model.load_from_checkpoint(checkpoint_path=args.vae_checkpoint_path)

        # Set VAE to evaluation mode
        vae_model = vae_model.eval()

        # Load VAE onto the correct device
        vae_model = vae_model.to(self.device)
        
        # Freeze the VAE to prevent its parameters from updating
        for param in vae_model.parameters():
            param.requires_grad = False
        
        # Wrap the VAE with accelerator (assuming you have an 'accelerator' attribute in your class)
        vae_model = self.accelerator.prepare(vae_model)

        return vae_model

    def find_last_ckpt(self):
        """
        Find the last checkpoint in the 'last' checkpoint directory.
        Returns a tuple (resume: bool, step: int).
        """
        last_ckpt_dir = os.path.join(self.ckpt_dir, 'last')
        if not os.path.exists(last_ckpt_dir):
            return False, 0

        ckpt_files = os.listdir(last_ckpt_dir)
        max_step = -1

        # Regex to extract the iteration number from the checkpoint filename
        pattern = re.compile(r'_(\d+)\.ckpt')

        for file in ckpt_files:
            match = pattern.search(file)
            if match:
                step = int(match.group(1))
                max_step = max(max_step, step)

        if max_step > -1:
            return True, max_step
        else:
            return False, 0

    def build_checkpoint(self):
        self.ckpt_dir = './checkpoints/'
        self.ckpt_prefixes = ["net", "sample_net", "optimizer"]
        
        # Create the checkpoint directory if it doesn't exist
        if self.accelerator.is_main_process:
            os.makedirs(self.ckpt_dir, exist_ok=True)

        if self.args.get('checkpoint_run', False):
            # Use specified checkpoints
            self.resume, self.step = True, self.args.checkpoint_iter
            print(f"Resuming training at iter {self.step}")
            self.checkpoint_b = hydra.utils.to_absolute_path(self.args.checkpoint_b)
            self.sample_checkpoint_b = hydra.utils.to_absolute_path(self.args.sample_checkpoint_b)
            self.optimizer_checkpoint_b = hydra.utils.to_absolute_path(self.args.optimizer_checkpoint_b)
        else:
            # Resume training from the last checkpoint, if it exists
            self.resume, self.step = self.find_last_ckpt()

            if self.resume:
                # Use the last checkpoint
                self.checkpoint_b, self.sample_checkpoint_b, self.optimizer_checkpoint_b = [
                    os.path.join(self.ckpt_dir, 'last', f"{ckpt_prefix}_{self.step:07}.ckpt")
                    for ckpt_prefix in self.ckpt_prefixes
                ]
        
        print(f'Resuming {self.resume}')

    def build_model(self):
        # Code to initialize the diffusion model
        model = get_model(self.args)
        if self.resume:
            try:
                model.load_state_dict(torch.load(self.checkpoint_b))
            except:
                state_dict = torch.load(self.checkpoint_b)
                new_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    name = k.replace("module.", "")  # remove "module."
                    new_state_dict[name] = v
                model.load_state_dict(new_state_dict)
        
        model = self.accelerator.prepare(model)
        self.model = model

    def remove_checkpoint(self, attribute_name):
        """Remove an existing checkpoint file."""
        if hasattr(self, attribute_name) and os.path.exists(getattr(self, attribute_name)):
            os.remove(getattr(self, attribute_name))

    def save_checkpoint(self, state_dict, filename, attribute_name):
        """Save a checkpoint and update its record."""
        checkpoint_path = os.path.join(self.ckpt_dir, 'last', filename)
        torch.save(state_dict, checkpoint_path)
        setattr(self, attribute_name, checkpoint_path)

    def save_last_ckpt(self, i):
        """Save the latest checkpoint and manage old ones."""
        # Create a directory for the last checkpoint if it doesn't exist
        last_ckpt_dir = os.path.join(self.ckpt_dir, 'last')
        if not os.path.exists(last_ckpt_dir):
            os.makedirs(last_ckpt_dir)

        # Proceed only if the current process is the main process in a distributed setting
        if self.accelerator.is_main_process:
            # Remove existing network and optimizer checkpoints
            self.remove_checkpoint('last_saved_net')
            self.remove_checkpoint('last_saved_opt')

            # Save the current network and optimizer checkpoints
            self.save_checkpoint(self.accelerator.unwrap_model(self.model).state_dict(), f'net_{i:07}.ckpt', 'last_saved_net')
            self.save_checkpoint(self.optimizer.optimizer.state_dict(), f'optimizer_{i:07}.ckpt', 'last_saved_opt')

            # Handle the saving of the EMA model if it's being used
            if self.args.ema:
                # Remove existing EMA model checkpoint
                self.remove_checkpoint('last_saved_net_ema')

                # Save the current EMA model checkpoint
                sample_net = self.ema_helper.ema_copy(self.accelerator.unwrap_model(self.model))
                self.save_checkpoint(sample_net.state_dict(), f'sample_net_{i:07}.ckpt', 'last_saved_net_ema')

    def accelerate(self,):
        (self.model, self.optimizer) = self.accelerator.prepare(
            self.model, self.optimizer)

    def update_ema(self, ):
        if self.args.ema:
            self.ema_helper = EMAHelper(mu=self.args.ema_rate, device=self.device)
            self.ema_helper.register(self.accelerator.unwrap_model(self.model))

    def build_ema(self):
        if self.args.ema:
            self.update_ema()
            if self.resume:
                sample_model = get_model(self.args)
                checkpoint_state_dict = torch.load(self.sample_checkpoint_b)
                sample_model.load_state_dict(checkpoint_state_dict)
                sample_model = sample_model.to(self.device)
                self.ema_helper.register(sample_model)
    
    def worker_init_fn(self, worker_id):
        np.random.seed(np.random.get_state()[1][0] + worker_id + self.accelerator.process_index * self.args.num_workers)

    def build_dataloader(self, ds, batch_size, shuffle=True, drop_last=True, repeat=True):
        dl_kwargs = {
            "num_workers": self.args.num_workers,
            "pin_memory": self.args.pin_memory,
            "worker_init_fn": self.worker_init_fn
        }

        dl = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, **dl_kwargs)
        dl = self.accelerator.prepare(dl)
        if repeat:
            dl = repeater(dl)
        return dl

    
    def compute_normalizing_constants(self):
        # Assuming 'encodings' is a tensor attribute of 'self.init_ds' that contains all the data
        # Flatten the encodings tensor to consider it as a concatenation of samples
        encodings = self.init_ds.encodings.view(-1)  # Flatten the tensor

        # Compute the mean and standard deviation directly from the flattened tensor
        mean = torch.mean(encodings)
        std_dev = torch.std(encodings)

        # Save the computed mean and standard deviation as attributes
        self.mean = mean
        self.std_dev = std_dev

        # Report the computed mean and standard deviation
        print(f'Mean: {self.mean}')
        print(f'Standard Deviation: {self.std_dev}')

        
    def build_dataloaders(self):
        self.plot_npar = min(self.args.plot_npar, len(self.init_ds))
        self.test_npar = min(self.args.test_npar, len(self.init_ds))

        self.cache_batch_size = min(self.args.cache_batch_size * self.accelerator.num_processes, len(self.init_ds))  # Adjust automatically to num_processes
        self.test_batch_size = min(self.args.test_batch_size * self.accelerator.num_processes, len(self.init_ds))    # Adjust automatically to num_processes

        self.init_dl = self.build_dataloader(self.init_ds, batch_size=self.batch_size // self.num_repeat_data)
        self.cache_init_dl = self.build_dataloader(self.init_ds, batch_size=self.cache_batch_size)

        self.save_init_dl = self.build_dataloader(self.init_ds, batch_size=self.test_batch_size, shuffle=False, repeat=False)
        self.save_dls_dict = {"train": self.save_init_dl}

        if self.valid_ds is not None:
            self.val_batch_size = self.batch_size // self.num_repeat_data
            self.val_total_batches = ceil(len(self.valid_ds) / self.val_batch_size)
            self.valid_dl = self.build_dataloader(self.valid_ds, batch_size=self.val_batch_size, shuffle=False)

            self.save_valid_dl = self.build_dataloader(self.valid_ds, batch_size=self.test_batch_size, shuffle=False, repeat=False)
            self.save_dls_dict["valid"] = self.save_valid_dl

        if self.test_ds is not None:
            self.save_test_dl = self.build_dataloader(self.test_ds, batch_size=self.test_batch_size, shuffle=False, repeat=False)
            self.save_dls_dict["test"] = self.save_test_dl

        if self.final_ds is not None:
            self.final_dl = self.build_dataloader(self.final_ds, batch_size=self.batch_size // self.num_repeat_data)
            self.cache_final_dl = self.build_dataloader(self.final_ds, batch_size=self.cache_batch_size)

            self.save_final_dl_repeat = self.build_dataloader(self.final_ds, batch_size=self.test_batch_size)
            self.save_final_dl = self.build_dataloader(self.final_ds, batch_size=self.test_batch_size, shuffle=False, repeat=False)
        else:
            self.final_dl = None
            self.cache_final_dl = None

            self.save_final_dl = None
            self.save_final_dl_repeat = None

        batch = next(self.cache_init_dl)
        batch_x = batch[0]
        batch_y = batch[1]

        shape_x = batch_x[0].shape
        self.shape_x = shape_x

        if isinstance(batch_y, list):
            if not batch_y:  # Check if the list is empty
                shape_y = ()
            else:
                shape_y = batch_y[0].shape
        elif torch.is_tensor(batch_y):
            if batch_y.nelement() == 0:  # Check if the tensor is empty
                shape_y = ()
            else:
                shape_y = batch_y.shape

        self.shape_y = shape_y

    def build_optimizer(self, forward_or_backward=None):
        optimizer = get_optimizer(self.model, self.args)
        if self.resume:
            optimizer.load_state_dict(torch.load(self.optimizer_checkpoint_b))
        self.optimizer = optimizer

    def get_sample_net(self,):
        if self.args.ema:
            sample_net = self.ema_helper.ema_copy(self.accelerator.unwrap_model(self.model))
        else:
            sample_net = self.model
        sample_net = sample_net.to(self.device)
        sample_net.eval()
        return sample_net

    def compute_current_step(self, i, n): #for compatibility with the SB plotting code.
        return i
    
    def plot_and_test_step(self, i, n, fb, sampler=None, datasets=['all'], num_steps='default'):
        self.set_seed(seed=0 + self.accelerator.process_index)
        test_metrics = self.plotter(i, n, fb, sampler='sde' if sampler is None else sampler, 
                                    datasets=datasets, num_steps=num_steps)
        if self.accelerator.is_main_process:
            self.save_logger.log_metrics(test_metrics, step=self.compute_current_step(i, n))
        return test_metrics

    def save_step(self, i):
        if (i+1) % self.stride == 0:
            self.save_last_ckpt(i)
            fb, n = 'b', 1
            for num_steps in self.test_num_steps_sequence:
                self.plot_and_test_step(i, n, fb, datasets=['train'], num_steps=num_steps)

    def get_named_beta_schedule(self, schedule_name, num_diffusion_timesteps): #helper fn for setting the SNR SDE
        """
        Get a pre-defined beta schedule for the given name.

        The beta schedule library consists of beta schedules which remain similar
        in the limit of num_diffusion_timesteps.
        Beta schedules may be added, but should not be removed or changed once
        they are committed to maintain backwards compatibility.
        """
        if schedule_name == "linear":
            # Linear schedule from Ho et al, extended to work for any number of
            # diffusion steps.
            scale = 1000 / num_diffusion_timesteps
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return np.linspace(
                beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
                )
        elif schedule_name == "cosine":
            return betas_for_alpha_bar(
                        num_diffusion_timesteps,
                        lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
                    )
        else:
            raise NotImplementedError(f"unknown beta schedule: {schedule_name}")
            
    def configure_sde(self, args):
        # Setup SDEs
        if args.sde.lower() == 'vpsde':
            self.sde = VPSDE(beta_min=args.beta_min, beta_max=args.beta_max, N=args.num_scales)
            self.sampling_eps = 1e-3
        elif args.sde.lower() == 'subvpsde':
            self.sde = subVPSDE(beta_min=args.beta_min, beta_max=args.beta_max, N=args.num_scales)
            self.sampling_eps = 1e-3
        elif args.sde.lower() == 'vesde':
            self.sde = VESDE(sigma_min=args.sigma_min, sigma_max=args.sigma_max, N=args.num_scales)
            self.sampling_eps = 1e-5
        elif args.sde.lower() == 'snrsde':
            self.sampling_eps = 1e-3
            if hasattr(args, 'beta_schedule'):
                # DISCRETE QUANTITIES
                N = args.num_scales
                betas = self.get_named_beta_schedule(args.beta_schedule, N)
                alphas = 1.0 - betas
                alphas_cumprod = np.cumprod(alphas, axis=0)
                discrete_snrs = alphas_cumprod/(1.0 - alphas_cumprod)

                # Monotonic Bicubic Interpolation
                snr = PchipInterpolator(np.linspace(self.sampling_eps, 1, len(discrete_snrs)), discrete_snrs)
                d_snr = snr.derivative(nu=1)

                def logsnr(t):
                    device = t.device
                    snr_val = torch.from_numpy(snr(t.cpu().numpy())).float().to(device)
                    return torch.log(snr_val)

                def d_logsnr(t):
                    device = t.device
                    dsnr_val = torch.from_numpy(d_snr(t.cpu().numpy())).float().to(device)
                    snr_val = torch.from_numpy(snr(t.cpu().numpy())).float().to(device)
                    return dsnr_val/snr_val

                self.sde = SNRSDE(N=N, gamma=logsnr, dgamma=d_logsnr)

            else:
                self.sde = SNRSDE(N=args.num_scales)
            
        else:
            raise NotImplementedError(f"SDE {config.training.sde} unknown.")
        
        self.sde.sampling_eps = self.sampling_eps

        if self.sde is None:
            raise ValueError("self.sde has not been initialized")

    def sample_batch(self, init_dl, final_dl):
        mean_final = self.mean_final
        std_final = self.std_final

        init_batch = next(init_dl)
        init_batch_x = init_batch[0]
        init_batch_y = init_batch[1]

        if self.final_ds is not None:
            final_batch = next(final_dl)
            final_batch_x = final_batch[0]
        else:
            mean_final = mean_final.to(init_batch_x.device)
            std_final = std_final.to(init_batch_x.device)
            final_batch_x = mean_final + std_final * torch.randn_like(init_batch_x)

        mean_final = mean_final.to(init_batch_x.device)
        std_final = std_final.to(init_batch_x.device)
        var_final = std_final ** 2

        if not self.cdsb:
            init_batch_y = None

        return init_batch_x, init_batch_y, final_batch_x, mean_final, var_final

    def get_score_fn(self, sde, model):
        def score_fn(x, y, t):
            labels = t * (sde.N - 1)
            noise_prediction = model.forward(x, y, labels)
            std = sde.marginal_prob(torch.zeros_like(x), t)[1]
            score = - noise_prediction / std[(...,)+(None,)*len(x.shape[1:])] #convert the noise prediction into a score prediction
            return score
        return score_fn

    def compute_loss(self, x, y, model): #we don't use the likelihood weighting
        #the model is a noise predictor and then we scale the output accordingly to convert the noise prediction into estimation of the score function
        score_fn = self.get_score_fn(self.sde, model) 
        eps = self.sampling_eps
        T = self.sde.T
        t = torch.rand(x.shape[0]).type_as(x) * (T - eps) + eps
        z = torch.randn_like(x)
        mean, std = self.sde.marginal_prob(x, t)
        perturbed_x = mean + std[(...,) + (None,) * len(x.shape[1:])] * z
        score = score_fn(perturbed_x, y, t) #score prediction
        losses = torch.square(score * std[(...,) + (None,) * len(x.shape[1:])] + z)
        losses = torch.mean(losses.reshape(losses.shape[0], -1), dim=-1)
        loss = torch.mean(losses)
        return loss

    def train(self):
        step = self.step
        # Set the random seed for reproducibility
        seed = ((step - 1) * self.accelerator.num_processes + self.accelerator.process_index) % 2**32
        self.set_seed(seed=seed)

        self.i = step - 1
        # Initialize the accelerator
        self.accelerate()

        num_iter = self.first_num_iter

        # Training loop
        for i in tqdm(range(step, num_iter + 1), mininterval=30):
            self.model.train()
            # Update the seed for each iteration
            seed = (i * self.accelerator.num_processes + self.accelerator.process_index) % 2**32
            self.set_seed(seed=seed)

            # Sample a batch for training
            x0, y, _, _, _ = self.sample_batch(self.init_dl, self.final_dl)

            if self.args.space == 'latent':
                # Assuming self.mean is the mean of x0 and has been computed previously
                x0 = (x0 - self.mean) / self.std_dev

            x0 = x0.to(self.device)
            # Repeat the data for augmentation, if applicable
            x0 = x0.repeat_interleave(self.num_repeat_data, dim=0)
            if self.cdsb:
                y = y.to(self.device)
                y = y.repeat_interleave(self.num_repeat_data, dim=0)

            # Compute the loss
            loss = self.compute_loss(x0, y, model=self.model)

            # Backward pass
            self.accelerator.backward(loss)
            # Gradient clipping, if enabled
            if self.grad_clipping:
                clipping_param = self.args.grad_clip
                total_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), clipping_param)
            else:
                total_norm = 0.
            
            # Log metrics
            if i == 1 or i % self.stride_log == 0 or i == num_iter:
                self.logger.log_metrics({'loss': loss, 'grad_norm': total_norm}, step=i)
            
            # Update weights and reset gradients
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            # Update EMA model, if applicable
            if self.args.ema:
                self.ema_helper.update(self.accelerator.unwrap_model(self.model))

            self.i = i

            # Validation phase
            if self.valid_ds is not None and (i+1) % self.validation_stride == 0:
                # Save the last checkpoint
                self.save_last_ckpt(i)

                # Prepare EMA model for validation
                ema_model = self.get_sample_net()
                total_valid_loss = 0.0
                num_valid_batches = 0
                val_total_batches = ceil(len(self.valid_ds) / self.val_batch_size)
                
                with torch.no_grad():  # Disable gradient calculations for validation
                    for _ in range(val_total_batches):
                        # Sample a batch for validation
                        x0, y, _, _, _ = self.sample_batch(self.valid_dl, self.final_dl)
                        
                        if self.args.space == 'latent':
                            # Assuming self.mean is the mean of x0 and has been computed previously
                            x0 = (x0 - self.mean) / self.std_dev

                        x0 = x0.to(self.device)
                        x0 = x0.repeat_interleave(self.num_repeat_data, dim=0)
                        if self.cdsb:
                            y = y.to(self.device)
                            y = y.repeat_interleave(self.num_repeat_data, dim=0)
                        
                        # Compute validation loss
                        valid_loss = self.compute_loss(x0, y, ema_model)
                        total_valid_loss += valid_loss.item()
                        num_valid_batches += 1
                
                # Log validation metrics
                avg_valid_loss = total_valid_loss / num_valid_batches
                self.logger.log_metrics({'val_loss': avg_valid_loss}, step=i)

            self.save_step(i)

    def test(self):
        i=400000
        fb, n = 'b', 1
        for num_steps in self.test_num_steps_sequence:
            self.plot_and_test_step(i, n, fb, datasets=['train'], num_steps=num_steps)

    def backward_sample(self, final_batch_x, y_c, permute=True, num_steps=None, calc_energy=False):
        denoise=True

        if num_steps is None:
            num_steps = self.num_steps

        # Retrieve the model used for sampling (usually an Exponential Moving Average (EMA) model).
        sample_net = self.get_sample_net()
        sample_net.to(self.device)  # Move the model to the specified device (e.g., GPU).

        # Get the score function associated with the SDE and the sampling network.
        score_fn = self.get_score_fn(self.sde, sample_net)

        with torch.no_grad():  # Disable gradient computation for efficiency.
            # Convert the input batch to the appropriate device and data type.
            #final_batch_x: sample from the stationary distribution. 
            #make sure  final_batch_x is sampled from the prior distribution (usually N(0,I))
            x = final_batch_x.to(self.device) 

            # Generate timesteps linearly spaced between the SDE's end time and a small epsilon value.
            timesteps = torch.linspace(self.sde.T, self.sampling_eps, num_steps + 1, device=self.device)

            N = x.shape[0]  # Number of samples in the batch.
            # Initialize a tensor to store the total samples over all timesteps.
            x_tot = torch.Tensor(N, num_steps, *self.shape_x).to(x.device)
            y_tot = None  # Placeholder for future use, currently unused.
            # Initialize a tensor to store the expanded timesteps for each sample.
            steps_expanded = torch.Tensor(N, num_steps, 1).to(x.device)

            # Iterate over the number of steps in the backward sampling process.
            for i in tqdm(range(num_steps)):
                t = timesteps[i]  # Current time step.
                dt = timesteps[i + 1] - timesteps[i]  # Time interval between steps.

                # Vector of current time steps, expanded to match the batch size.
                vec_t = torch.ones(N, device=self.device) * t

                # Euler-Maruyama predictor step:
                # Calculate the drift and diffusion terms of the SDE at the current step.
                drift, diffusion = self.sde.sde(x, vec_t)
                # Compute the score (gradient of log probability) at the current step.
                score = score_fn(x, y_c, vec_t)
                # Calculate the drift of the reverse SDE.
                rdrift = drift - diffusion[(..., ) + (None, ) * len(x.shape[1:])] ** 2 * score
                # Generate random noise for the diffusion term.
                noise = torch.randn_like(x)

                # Update the samples using the Euler-Maruyama scheme.
                # When denoise=True, we use the mean at the last integration step
                x = x + rdrift * dt
                if not (denoise and (i==num_steps-1)):
                    x = x + diffusion[(..., ) + (None, ) * len(x.shape[1:])] * torch.sqrt(-dt) * noise
                
                # Store the updated samples and the corresponding time steps.
                x_tot[:, i, ::] = x
                steps_expanded[:, i, :] = t

        if self.args.space == 'latent':
            # Revert the standardization of the last point of the trajectory
            x_tot[:, -1, ...] = (x_tot[:, -1, ...] * self.std_dev) + self.mean


        # Return the total trajectory, along with other additional information (currently placeholders).
        # sample: the last point of the trajectory, i.e., sample = x_tot[-1].
        return x_tot, num_steps, [], []