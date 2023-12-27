import torch
from omegaconf import OmegaConf
import hydra
from ..models import *
from ..data import load_dataset_filenames, create_dataset
from .plotters import *
import torchvision.datasets
import torchvision.transforms as transforms
import os
from functools import partial
from .logger import CSVLogger, WandbLogger, Logger
from torch.utils.data import DataLoader, random_split
from bridge.data.downscaler import DownscalerDataset
import lpips
from pytorch_lightning.callbacks import ModelCheckpoint
import wandb
import torch.utils.data as data
import pytorch_lightning as pl
from tqdm import tqdm

cmp = lambda x: transforms.Compose([*x])

def worker_init_fn(worker_id):
    np.random.seed(worker_id)
    torch.manual_seed(worker_id)
    torch.cuda.manual_seed_all(worker_id)


def get_plotter(runner, args, fid_feature=2048):
    is_SB = args.get('is_SB', True)
    if args.space == 'latent':
        return LatentPlotter(runner, args, is_SB=is_SB)
    else:
        dataset_tag = getattr(args, DATASET)
        if dataset_tag in [DATASET_MNIST, DATASET_EMNIST, DATASET_CIFAR10]:
            return ImPlotter(runner, args, fid_feature=fid_feature, is_SB=is_SB)
        elif dataset_tag in [DATASET_DOWNSCALER_LOW, DATASET_DOWNSCALER_HIGH]:
            return DownscalerPlotter(runner, args, is_SB=is_SB)
        else:
            return Plotter(runner, args, is_SB=is_SB)


# Model
# --------------------------------------------------------------------------------

MODEL = 'Model'
BASIC_MODEL = 'Basic'
UNET_MODEL = 'UNET'
DOWNSCALER_UNET_MODEL = 'DownscalerUNET'
MLP_MODEL = 'MLP'

NAPPROX = 2000

def get_model(args):
    if args.space == 'latent':
        model_type = getattr(args, MODEL)
        if model_type == 'MLP':
            latent_dim = args.latent_dim

            layer_widths = args.model.inter_layer_widths.copy()
            layer_widths.append(latent_dim)

            net = MLPScoreNetwork(input_dim=latent_dim+1,
                            layer_widths=layer_widths,
                            activate_final=False)
        elif model_type == 'MLPSkipNet':
            skip_layers = list(range(args.model.starting_skip_layer, args.model.num_layers))
            kwargs = {
                'num_channels':args.model.in_channels,
                'num_layers':args.model.num_layers,
                'skip_layers': skip_layers, 
                'num_hid_channels': args.model.num_hid_channels, 
                'num_time_emb_channels': args.model.num_time_emb_channels, 
                'activation': Activation.silu, 
                'use_norm': args.model.use_norm, 
                'condition_bias': args.model.condition_bias, 
                'dropout': args.model.dropout, 
                'last_act': Activation.none, 
                'num_time_layers': args.model.num_time_layers, 
                'time_last_act': args.model.time_last_act
            }
            net = MLPSkipNet(**kwargs)
        elif model_type == 'UNET':
            print(args.model.channel_mult)
            num_encoder_resolutions = len(args.encoder.ch_mult)
            observation_image_size = args.data.image_size
            latent_image_size = observation_image_size // 2**(num_encoder_resolutions-1)
            attention_ds = []
            for res in args.model.attention_resolutions:
                if latent_image_size % res == 0:
                    attention_ds.append(latent_image_size // res)

            kwargs = {
                "in_channels": args.encoder.z_channels,
                "model_channels": args.model.num_channels,
                "out_channels": args.encoder.z_channels,
                "num_res_blocks": args.model.num_res_blocks,
                "attention_resolutions": tuple(attention_ds),
                "dropout": args.model.dropout,
                "channel_mult": args.model.channel_mult,
                "num_classes": None,
                "use_checkpoint": args.model.use_checkpoint,
                "num_heads": args.model.num_heads,
                "use_scale_shift_norm": args.model.use_scale_shift_norm,
                "resblock_updown": args.model.resblock_updown,
                "temb_scale": args.model.temb_scale
            }

            net = UNetModel(**kwargs)

        elif model_type == 'LatentUNET':
            net = LatentUNET(args)
    else:
        model_tag = getattr(args, MODEL)

        if model_tag == UNET_MODEL:
            image_size = args.data.image_size

            if args.model.channel_mult is not None:
                channel_mult = args.model.channel_mult
                # Check if channel_mult is already a tuple
                if not isinstance(channel_mult, tuple):
                    channel_mult=tuple(channel_mult)
            else:
                #default channel multipliers for fixed resolutions
                if image_size == 256:
                    channel_mult = (1, 1, 2, 2, 4, 4)
                elif image_size == 160:
                    channel_mult = (1, 2, 2, 4)
                elif image_size == 64:
                    channel_mult = (1, 2, 2, 2)
                elif image_size == 32:
                    channel_mult = (1, 2, 2, 2)
                elif image_size == 28:
                    channel_mult = (0.5, 1, 1)
                else:
                    raise ValueError(f"unsupported image size: {image_size}")

            attention_ds = []
            for res in args.model.attention_resolutions.split(","):
                if image_size % int(res) == 0:
                    attention_ds.append(image_size // int(res))

            kwargs = {
                "in_channels": args.data.channels,
                "model_channels": args.model.num_channels,
                "out_channels": args.data.channels,
                "num_res_blocks": args.model.num_res_blocks,
                "attention_resolutions": tuple(attention_ds),
                "dropout": args.model.dropout,
                "channel_mult": channel_mult,
                "num_classes": None,
                "use_checkpoint": args.model.use_checkpoint,
                "num_heads": args.model.num_heads,
                "use_scale_shift_norm": args.model.use_scale_shift_norm,
                "resblock_updown": args.model.resblock_updown,
                "temb_scale": args.model.temb_scale
            }

            net = UNetModel(**kwargs)
        elif model_tag == DOWNSCALER_UNET_MODEL:
            image_size = args.data.image_size
            channel_mult = args.model.channel_mult

            kwargs = {
                "in_channels": args.data.channels,
                "cond_channels": args.data.cond_channels, 
                "model_channels": args.model.num_channels,
                "out_channels": args.data.channels,
                "num_res_blocks": args.model.num_res_blocks,
                "dropout": args.model.dropout,
                "channel_mult": channel_mult,
                "temb_scale": args.model.temb_scale, 
                "mean_bypass": args.model.mean_bypass,
                "scale_mean_bypass": args.model.scale_mean_bypass,
                "shift_input": args.model.shift_input,
                "shift_output": args.model.shift_output,
            }

            net = DownscalerUNetModel(**kwargs)
            
    return net

class VAE(pl.LightningModule):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.latent_dim = config.latent_dim
            net_options = {
                'simple_encoder': ConvNetEncoder,
                'simple_decoder': ConvNetDecoder,
                'deep_encoder': DeepEncoder,
                'deep_decoder': DeepDecoder,
            }
            self.encoder = net_options[config.encoder.name](config)
            self.decoder = net_options[config.decoder.name](config)
            self.decoder_tanh = nn.Tanh()
            self.save_hyperparameters()

        def forward(self, x):
            return x

        def encode(self, x):
            if self.config.variational:
                mean_z, log_var_z = self.encoder(x)
                return mean_z, log_var_z
            else:
                z = self.encoder(x)
                return z, None

        def decode(self, z):
            out = self.decoder(z)
            # Use tanh for normalization of output to range [-1,1].
            mean_x = self.decoder_tanh(out)
            return mean_x, None

        def reconstruct(self, x):
            mean_z, log_var_z = self.encode(x)
            
            if isinstance(log_var_z, torch.Tensor):
                z = torch.randn_like(mean_z, device=self.device) * torch.sqrt(log_var_z.exp()) + mean_z
            else:
                z = mean_z
                
            mean_x, _ = self.decode(z)
            return mean_x

        
        def sample(self, num_samples):
            # Check if the decoder has the attribute 'z_shape'
            if hasattr(self.decoder, 'z_shape'):
                z_shape = self.decoder.z_shape
                # Sample z with shape (num_samples, z_shape[1], z_shape[2], z_shape[3])
                z = torch.randn((num_samples,) + z_shape[1:], device=self.device)
            else:
                # If 'z_shape' is not an attribute, use the latent_dim
                z = torch.randn((num_samples, self.latent_dim), device=self.device)

            # Decode the latent samples to get generated samples
            mean_x, _ = self.decode(z)
            return mean_x

            
        def handle_batch(self, batch):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            return batch 

        def compute_loss(self, batch):
            if self.global_step < 5:
                assert batch.min() >= -1 and batch.max() <= 1, "Data should be normalized in the range [-1,1] because we use a tanh layer in the decoder"
            
            B = batch.shape[0]
            if self.config.variational:
                mean_z, log_var_z = self.encode(batch)  # Output shape could be (B, z_shape) or (B, z_channels, res, res)
                z = torch.randn_like(mean_z, device=self.device) * torch.sqrt(log_var_z.exp()) + mean_z
                mean_x, _ = self.decode(z)

                # Flatten the tensors if they are 2D
                flat_mean_z = mean_z.view(B, -1)
                flat_log_var_z = log_var_z.view(B, -1)

                kl_loss = -0.5 * torch.sum(1 + flat_log_var_z - flat_mean_z ** 2 - flat_log_var_z.exp(), dim=1)
                kl_loss = kl_loss.mean()
                
                rec_loss = torch.linalg.norm(batch.view(B, -1) - mean_x.view(B, -1), dim=1)
                rec_loss = rec_loss.mean()
                kl_weight = self.config.kl_weight
                loss = rec_loss + kl_weight * kl_loss
            else:
                z, _ = self.encode(batch)
                mean_x, _ = self.decode(z)
                rec_loss = torch.linalg.norm(batch.view(B, -1) - mean_x.view(B, -1), dim=1)
                rec_loss = rec_loss.mean()
                kl_loss = torch.tensor(0., device=self.device)
                loss = rec_loss

            return loss, rec_loss.detach(), kl_loss.detach()

        
        def training_step(self, batch, batch_idx):
            batch = self.handle_batch(batch)
            loss, rec_loss, kl_loss = self.compute_loss(batch)
            self.log('train_loss', loss, sync_dist=True)
            self.log('train_rec_loss', rec_loss, sync_dist=True)
            self.log('train_kl_loss', kl_loss, sync_dist=True)
            return loss
        
        def validation_step(self, batch, batch_idx):
            batch = self.handle_batch(batch)
            loss, rec_loss, kl_loss = self.compute_loss(batch)
            self.log('val_loss', loss, sync_dist=True)
            self.log('val_rec_loss', rec_loss, sync_dist=True)
            self.log('val_kl_loss', kl_loss, sync_dist=True)
            return loss
        
        def test_step(self, batch, batch_idx):
            batch = self.handle_batch(batch)

            if batch_idx == 0:
                self.lpips_distance_fn = lpips.LPIPS(net='vgg').to(self.device) 
                
            z, _ = self.encode(batch)
            reconstruction, _ = self.decode(z)

            avg_lpips_score = torch.mean(self.lpips_distance_fn(reconstruction, batch))

            difference = torch.flatten(reconstruction, start_dim=1)-torch.flatten(batch, start_dim=1)
            L2norm = torch.linalg.vector_norm(difference, ord=2, dim=1)
            avg_L2norm = torch.mean(L2norm)

            self.log("LPIPS", avg_lpips_score, on_step=True, on_epoch=True, prog_bar=True, logger=True)
            self.log("L2", avg_L2norm, on_step=True, on_epoch=True, prog_bar=True, logger=True)

            output = dict({
            'LPIPS': avg_lpips_score,
            'L2': avg_L2norm,
            })
            return output
        
        def configure_optimizers(self):
            optim = torch.optim.Adam(self.parameters(), lr=self.config.vae_optim.lr)
            if self.config.vae_optim.use_scheduler:
                sch = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 
                                                                factor=self.config.vae_optim.sch_factor, 
                                                                patience=self.config.vae_optim.sch_patience, 
                                                                min_lr=self.config.vae_optim.sch_min_lr)
                return {'optimizer': optim, 
                        "lr_scheduler" : {
                            "scheduler" : sch,
                            "monitor" : "val_loss",
                        }
                    }
            else:
                return optim

def get_vae_model(args):    
    vae_model = VAE(args)
    return vae_model

def create_stochastic_vae_encodings_dataset(vae_model, base_dataset, batch_size, dataset='train'):
    stochastic_encodings = []
    dataloader = DataLoader(base_dataset, batch_size=batch_size, shuffle=False)

    pbar = tqdm(total=len(base_dataset), desc=f"Encoding {dataset} dataset")

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            batch = vae_model.handle_batch(batch)
            batch = batch.to(vae_model.device)
            mean_z, log_var_z = vae_model.encode(batch)

            if isinstance(log_var_z, torch.Tensor):
                assert vae_model.config.variational==True
                # Concatenate mean and log variance
                concatenated = torch.cat((mean_z, log_var_z), dim=1)
            else:
                assert vae_model.config.variational==False
                concatenated = mean_z

            '''
            #check reconstruction
            split_index = concatenated.size(1) // 2
            mean, log_var = concatenated[:, :split_index], concatenated[:, split_index:]
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sampled_z = eps.mul(std).add_(mean)
            decoded_image, _ = vae_model.decode(sampled_z)
            # Compute the L2 distance for each image in the batch
            l2_distances = torch.sqrt(torch.sum((batch - decoded_image) ** 2, dim=[1, 2, 3]))
            # Compute the average L2 distance over all images in the batch
            average_l2_distance = torch.mean(l2_distances)
            print(f'average_l2_distance: {average_l2_distance.item()}')
            '''

            stochastic_encodings.extend(concatenated.cpu().numpy())

            pbar.update(batch_size)

    pbar.close()

    return stochastic_encodings

def compute_wasserstein_distance(self, mean1, cov1, mean2, cov2):
    def matrix_sqrt(self, mat):
        U, S, V = torch.svd(mat)
        return U @ torch.diag(torch.sqrt(S)) @ V.t()

    diff_mean = mean1 - mean2
    sqrt_cov1 = matrix_sqrt(cov1)
    sqrt_cov2 = matrix_sqrt(cov2)
        
    # Squared Euclidean distance between the means
    term1 = torch.norm(diff_mean, p=2)**2
        
    # The trace term in the Wasserstein distance formula
    term2 = torch.trace(cov1 + cov2 - 2 * matrix_sqrt(sqrt_cov1 @ cov2 @ sqrt_cov1))
        
    # Compute the Wasserstein distance
    w2_distance = term1 + term2
    return w2_distance.item()

def create_sampled_dataset_faster(mean_var):
    mean_z, log_var_z = torch.split(mean_var, mean_var.size(1) // 2, dim=1)
    std_z = torch.exp(0.5 * log_var_z)
    eps = torch.randn_like(log_var_z)
    sampled_z = eps.mul(std_z).add_(mean_z)
    return sampled_z

def create_sampled_dataset(data_samples, mean_var_dataset, num_samples=1):
    sampled_dataset = []

    print(mean_var_dataset.size())
    with torch.no_grad():
        for mean_var in mean_var_dataset[:data_samples]:
            mean_var = mean_var.unsqueeze(0)
            #print("mean_var shape:", mean_var.shape)
            mean_z, log_var_z = torch.split(mean_var, mean_var.size(1) // 2, dim=1)

            mean_z = mean_z.repeat(num_samples, 1)
            std_z = torch.exp(0.5 * log_var_z)
            eps = torch.randn_like(std_z).repeat(num_samples, 1)
            sampled_z = eps.mul(std_z).add_(mean_z)

            sampled_dataset.extend(sampled_z.cpu().numpy())


    return sampled_dataset


def calculate_mean_covariance(sampled_dataset):
    # Reshape the dataset from (batch, 3, w, h) to (batch, 3*w*h)
    reshaped_dataset = sampled_dataset.reshape(sampled_dataset.shape[0], -1)
    
    # Calculate the mean and covariance on the reshaped dataset
    mean = np.mean(reshaped_dataset, axis=0)
    covariance = np.cov(reshaped_dataset, rowvar=False)
    
    return mean, covariance

def print_dataset_stats(sampled_mean, sampled_covariance):
    # Calculate the mean and std of values in the first diagonal and the two adjacent diagonals
    diagonal_values = np.diag(sampled_covariance)
    adjacent_values1 = np.diag(sampled_covariance, k=1)
    adjacent_values2 = np.diag(sampled_covariance, k=-1)

    diagonal_mean = np.mean(diagonal_values)
    diagonal_std = np.std(diagonal_values)
    adjacent_mean1 = np.mean(adjacent_values1)
    adjacent_std1 = np.std(adjacent_values1)
    adjacent_mean2 = np.mean(adjacent_values2)
    adjacent_std2 = np.std(adjacent_values2)

    # Calculate the mean and std of the means of the sampled dataset
    mean_of_sampled_means = np.mean(sampled_mean)
    std_of_sampled_means = np.std(sampled_mean)

    # Print the results
    print("Mean of Diagonal Values:", diagonal_mean)
    print("Std of Diagonal Values:", diagonal_std)
    print("Mean of Adjacent Diagonal Values (k=1):", adjacent_mean1)
    print("Std of Adjacent Diagonal Values (k=1):", adjacent_std1)
    print("Mean of Adjacent Diagonal Values (k=-1):", adjacent_mean2)
    print("Std of Adjacent Diagonal Values (k=-1):", adjacent_std2)
    print("Mean of Sampled Means:", mean_of_sampled_means)
    print("Std of Sampled Means:", std_of_sampled_means)


class VAEEncodingsDataset(data.Dataset):
    def __init__(self, encodings, variational):
        # Convert encodings to tensor once during initialization
        self.encodings = torch.tensor(encodings)
        self.variational = variational

    def __len__(self):
        return len(self.encodings)

    def __getitem__(self, idx):
        encoding = self.encodings[idx]
        
        if self.variational:
            # Split the encoding into mean and log variance
            split_index = encoding.size(0) // 2
            mean, log_var = encoding[:split_index], encoding[split_index:]

            # Reparameterization trick to sample
            std = torch.exp(0.5 * log_var)
            eps = torch.randn_like(std)
            sampled_z = eps.mul(std).add_(mean)
        else:
            sampled_z = encoding

        return sampled_z, []

def generate_unconditional_samples(vae_model, base_dataset, batch_size):
    unconditional_samples = []
    num_samples = len(base_dataset)
    
    # Initialize tqdm for progress bar
    pbar = tqdm(total=num_samples, desc="Generating samples")

    # Turn off gradient tracking
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            num_samples_this_batch = min(batch_size, num_samples - i)
            # Get the unconditional samples from the VAE
            x_samples = vae_model.sample(num_samples_this_batch)
            unconditional_samples.extend(x_samples.to('cpu'))

            # Update the tqdm progress bar
            pbar.update(num_samples_this_batch)

    pbar.close()
    return unconditional_samples

class VAEUnconditionalDataset(data.Dataset):
    def __init__(self, unconditional_samples):
        self.unconditional_samples = unconditional_samples

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.unconditional_samples)

    def __getitem__(self, idx):
        return self.unconditional_samples[idx], []

#checkpoint callback for the vae
def get_checkpoint_callback(logger):
    # For WandbLogger, leverage the default wandb run directory for saving checkpoints
    if isinstance(logger, WandbLogger):
        dirpath = os.path.join(wandb.run.dir, 'vae_checkpoints')
    else:
        # Otherwise, use your custom path or a default one
        dirpath = os.path.join("logs", "checkpoints")

    checkpoint_callback = ModelCheckpoint(
        dirpath=dirpath,
        monitor='val_loss',
        filename='{epoch}--{val_loss:.3f}',
        save_last=True,
        save_top_k=3
    )

    return checkpoint_callback
        
# Optimizer
# --------------------------------------------------------------------------------

def get_optimizer(net, args):
    lr = args.lr
    optimizer = args.optimizer
    if optimizer == 'Adam':
        return torch.optim.Adam(net.parameters(), lr=lr)
    elif optimizer == 'AdamW':
        return torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=args.weight_decay)


# Dataset
# --------------------------------------------------------------------------------

DATASET = 'Dataset'
DATASET_TRANSFER = 'Dataset_transfer'
DATASET_MNIST = 'mnist'
DATASET_EMNIST = 'emnist'
DATASET_CIFAR10 = 'cifar10'
DATASET_DOWNSCALER_LOW = 'downscaler_low'
DATASET_DOWNSCALER_HIGH = 'downscaler_high'
DATASET_VAE = 'vae'

def get_datasets(args, device=None):
    dataset_tag = getattr(args, DATASET)

    # INITIAL (DATA) DATASET

    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)
    print(data_dir)

    # MNIST DATASET
    if dataset_tag == DATASET_MNIST:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'mnist')
        load = args.load
        assert args.data.channels == 1
        assert args.data.image_size == 28
        train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        init_ds = torchvision.datasets.MNIST(root=root, train=True, transform=cmp(train_transform), download=True)

    # CIFAR10 DATASET
    elif dataset_tag == DATASET_CIFAR10:
        root = os.path.join(data_dir, 'cifar10')
        load = args.load
        assert args.data.channels == 3
        assert args.data.image_size == 32
        train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        if args.data.random_flip:
            train_transform.insert(0, transforms.RandomHorizontalFlip())
            
        init_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=cmp(train_transform), download=True)

    elif dataset_tag in ['ffhq']:
        root = os.path.join(data_dir, dataset_tag)
        percentage_use = args.data.percentage_use
        train_percentage = args.data.train_percentage
        val_percentage = args.data.val_percentage
        random_flip = args.data.random_flip
        image_size = args.data.image_size
        random_crop = args.data.random_crop
        file_paths = load_dataset_filenames(root, train=True, percentage_use=percentage_use, train_percentage=train_percentage, val_percentage=val_percentage)
        init_ds = create_dataset(file_paths, random_crop, image_size, random_flip)
    
    #elif dataset_tag == DATASET_CIFAR10:
    #handle the case of the encoded distribution for all datasets.
    #set transfer to False in the dataset subconfig. so that the final_ds is the Normal distribution.

    # Downscaler dataset
    elif dataset_tag == DATASET_DOWNSCALER_HIGH:
        root = os.path.join(data_dir, 'downscaler')
        train_transform = [transforms.Normalize((0.,), (1.,))]
        assert not args.data.random_flip
        # if args.data.random_flip:
        #     train_transform = train_transform + [
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.RandomVerticalFlip(p=0.5),
        #         transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
        #     ]
        wavenumber = args.data.get('wavenumber', 0)
        split = args.data.get('split', "train")
        
        init_ds = DownscalerDataset(root=root, resolution=512, wavenumber=wavenumber, split=split, transform=cmp(train_transform))

    if args.space == 'latent':
        print('We are loading the latent encodings')
        #VAE setup 
        vae_model = VAE(args)
        vae_model = vae_model.load_from_checkpoint(args.vae_checkpoint_path)
        vae_model = vae_model.to(device)
        vae_model.eval()
                
        stochastic_encodings = create_stochastic_vae_encodings_dataset(vae_model, init_ds, args.batch_size, dataset='train')
        
        # Convert the list to a NumPy array
        stochastic_encodings = np.array(stochastic_encodings)
        print(stochastic_encodings.shape)

        sample_mean, sample_cov = calculate_mean_covariance(stochastic_encodings)
        print_dataset_stats(sample_mean, sample_cov)

        #from os import getcwd
        # Save the NumPy array in the current working directory
        #filename = "stochastic_encodings.npy"
        #current_directory = os.getcwd()
        #save_path = os.path.join(current_directory, filename)
        #np.save(save_path, stochastic_encodings_array)

        #print(f"Stochastic encodings saved to {save_path}")

        #dataset = create_sampled_dataset_faster(torch.tensor(stochastic_encodings)) #create_sampled_dataset(10000, torch.tensor(stochastic_encodings), 2)
        #sample_mean, sample_cov = calculate_mean_covariance(dataset)
        #print_dataset_stats(sample_mean, sample_cov)
        
        init_ds = VAEEncodingsDataset(stochastic_encodings, args.variational)

    # FINAL DATASET
    final_ds, mean_final, var_final = get_final_dataset(args, init_ds, device)
    return init_ds, final_ds, mean_final, var_final


def get_final_dataset(args, init_ds, device):
    if args.space == 'latent':
        if hasattr(args.encoder, 'z_channels'):
            num_encoder_resolutions = len(args.encoder.ch_mult)
            observation_image_size = args.data.image_size
            latent_image_size = observation_image_size // 2**(num_encoder_resolutions-1)
            z_channels = args.encoder.z_channels
            shape = [z_channels, latent_image_size, latent_image_size]
        else:
            latent_dim = args.latent_dim
            shape = [latent_dim,]
        
        mean_final = torch.zeros(shape)
        var_final = torch.ones(shape)
        final_ds = None
    
    else: #space=='observation'
        if args.transfer:
            data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)
            dataset_transfer_tag = getattr(args, DATASET_TRANSFER)
            mean_final = torch.tensor(0.)
            var_final = torch.tensor(1.*10**3)  # infty like

            if dataset_transfer_tag == DATASET_EMNIST:
                from ..data.emnist import FiveClassEMNIST
                # data_tag = args.data.dataset
                root = os.path.join(data_dir, 'emnist')
                load = args.load
                assert args.data.channels == 1
                assert args.data.image_size == 28
                train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
                final_ds = FiveClassEMNIST(root=root, train=True, download=True, transform=cmp(train_transform))

            elif dataset_transfer_tag == DATASET_DOWNSCALER_LOW:
                root = os.path.join(data_dir, 'downscaler')
                train_transform = [transforms.Normalize((0.,), (1.,))]
                if args.data.random_flip:
                    train_transform = train_transform + [
                        transforms.RandomHorizontalFlip(p=0.5),
                        transforms.RandomVerticalFlip(p=0.5),
                        transforms.RandomApply([transforms.RandomRotation((90, 90))], p=0.5),
                    ]

                split = args.data.get('split', "train")
                
                final_ds = DownscalerDataset(root=root, resolution=64, split=split, transform=cmp(train_transform))
            elif dataset_transfer_tag == DATASET_VAE:
                vae_model = VAE(args)
                vae_model = vae_model.load_from_checkpoint(args.vae_checkpoint_path)
                vae_model = vae_model.to(device)
                vae_model.eval()
                
                unconditional_samples = generate_unconditional_samples(vae_model, init_ds, args.batch_size)
                final_ds = VAEUnconditionalDataset(unconditional_samples)
        else:
            if args.adaptive_mean:
                vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, num_workers=args.num_workers, worker_init_fn=worker_init_fn)))[0]
                mean_final = vec.mean(axis=0)
                var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
            elif args.final_adaptive:
                vec = next(iter(DataLoader(init_ds, batch_size=NAPPROX, num_workers=args.num_workers, worker_init_fn=worker_init_fn)))[0]
                mean_final = vec.mean(axis=0)
                var_final = vec.var(axis=0)
            else:
                mean_final = eval(args.mean_final) if isinstance(args.mean_final, str) else torch.tensor([args.mean_final])
                var_final = eval(args.var_final) if isinstance(args.var_final, str) else torch.tensor([args.var_final])
            final_ds = None

    return final_ds, mean_final, var_final


def get_valid_test_datasets(args, device=None):
    valid_ds, test_ds = None, None

    dataset_tag = getattr(args, DATASET)
    data_dir = hydra.utils.to_absolute_path(args.paths.data_dir_name)

    # MNIST DATASET
    if dataset_tag == DATASET_MNIST:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'mnist')
        load = args.load
        assert args.data.channels == 1
        assert args.data.image_size == 28
        test_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        valid_ds = None
        test_ds = torchvision.datasets.MNIST(root=root, train=False, transform=cmp(test_transform), download=True)
    
    # CIFAR10 DATASET
    if dataset_tag == DATASET_CIFAR10:
        root = os.path.join(data_dir, 'cifar10')
        test_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        full_test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=transforms.Compose(test_transform), download=True)

        # Split the CIFAR10 test dataset into validation and test datasets
        valid_ds_size = len(full_test_ds) // 2
        test_ds_size = len(full_test_ds) - valid_ds_size
        valid_ds, test_ds = random_split(full_test_ds, [valid_ds_size, test_ds_size])
    
    elif dataset_tag in ['ffhq']:
        root = os.path.join(data_dir, dataset_tag)
        percentage_use = args.data.percentage_use
        train_percentage = args.data.train_percentage
        val_percentage = args.data.val_percentage
        random_flip = args.data.random_flip
        image_size = args.data.image_size
        random_crop = args.data.random_crop
        
        val_paths, test_paths = load_dataset_filenames(root, train=False, percentage_use=percentage_use, train_percentage=train_percentage, val_percentage=val_percentage)
        valid_ds = create_dataset(val_paths, random_crop, image_size, random_flip)
        test_ds = create_dataset(test_paths, random_crop, image_size, random_flip)

    if args.space == 'latent':
        #VAE setup 
        vae_model = VAE(args)
        vae_model = vae_model.load_from_checkpoint(args.vae_checkpoint_path)
        vae_model = vae_model.to(device)
        vae_model.eval()
        
        if valid_ds is not None:
            stochastic_encodings = create_stochastic_vae_encodings_dataset(vae_model, valid_ds, args.batch_size, dataset='validation')
            valid_ds = VAEEncodingsDataset(stochastic_encodings, args.variational)
        
        if test_ds is not None:
            stochastic_encodings = create_stochastic_vae_encodings_dataset(vae_model, test_ds, args.batch_size, dataset='test')
            test_ds = VAEEncodingsDataset(stochastic_encodings, args.variational)

    return valid_ds, test_ds


# Logger
# --------------------------------------------------------------------------------

LOGGER = 'LOGGER'
CSV_TAG = 'CSV'
WANDB_TAG = 'Wandb'
NOLOG_TAG = 'NONE'


def get_logger(args, name, project_name=None):
    logger_tag = getattr(args, LOGGER)

    if logger_tag == CSV_TAG:
        kwargs = {'save_dir': args.CSV_log_dir, 'name': name, 'flush_logs_every_n_steps': 1}
        return CSVLogger(**kwargs)

    if logger_tag == WANDB_TAG:
        log_dir = os.getcwd()
        
        if args.run_name is None:
            if not args.use_default_wandb_name:
                run_name = os.path.normpath(os.path.relpath(log_dir, os.path.join(
                    hydra.utils.to_absolute_path(args.paths.experiments_dir_name), args.name))).replace("\\", "/")
            else:
                run_name = None
        else:
            run_name = args.run_name
        
        data_tag = args.data.dataset
        config = OmegaConf.to_container(args, resolve=True)

        if 'WANDB_ENTITY' in os.environ:
            wandb_entity = os.environ['WANDB_ENTITY']
        else:
            wandb_entity = os.environ.get('WANDB_ENTITY', 'diffteam')

        assert len(wandb_entity) > 0, "WANDB_ENTITY not set"

        if project_name is None:
            project_name = 'dsbm_' + args.name

        kwargs = {'name': run_name, 'project': project_name, 'prefix': name, 'entity': wandb_entity,
                  'tags': [data_tag], 'config': config, 'id': str(args.wandb_id) if args.wandb_id is not None else None}
        return WandbLogger(**kwargs)

    if logger_tag == NOLOG_TAG:
        return Logger()
