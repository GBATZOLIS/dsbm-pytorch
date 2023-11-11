import torch
from omegaconf import OmegaConf
import hydra
from ..models import *
from .plotters import *
import torchvision.datasets
import torchvision.transforms as transforms
import os
from functools import partial
from .logger import CSVLogger, WandbLogger, Logger
from torch.utils.data import DataLoader
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
    dataset_tag = getattr(args, DATASET)
    if dataset_tag in [DATASET_MNIST, DATASET_EMNIST, DATASET_CIFAR10]:
        return ImPlotter(runner, args, fid_feature=fid_feature)
    elif dataset_tag in [DATASET_DOWNSCALER_LOW, DATASET_DOWNSCALER_HIGH]:
        return DownscalerPlotter(runner, args)
    else:
        return Plotter(runner, args)


# Model
# --------------------------------------------------------------------------------

MODEL = 'Model'
BASIC_MODEL = 'Basic'
UNET_MODEL = 'UNET'
DOWNSCALER_UNET_MODEL = 'DownscalerUNET'

NAPPROX = 2000


def get_model(args):
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
                'simple_decoder': ConvNetDecoder
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
            # Sample from a standard Gaussian distribution
            z = torch.randn((num_samples, self.latent_dim), device=self.device)
            # Decode the latent samples to get generated samples
            mean_x, _ = self.decode(z)
            return mean_x

        def compute_loss(self, batch):
            if self.global_step < 5:
                assert batch.min() >= -1 and batch.max() <= 1, "Data should be normalized in the range [-1,1] because we use a tanh layer in the decoder"
            
            B = batch.shape[0]
            D_Z = self.latent_dim
            if self.config.variational:
                mean_z, log_var_z = self.encode(batch) # (B, D_Z), (B, D_Z) assuming Sigma_z is diagonal
                z = torch.randn((B, self.latent_dim), device=self.device) * torch.sqrt(log_var_z.exp()) + mean_z
                mean_x, _ = self.decode(z) # (B, D_X), (B, D_X) assuming Sigma_x is the identity matrix. (fine when using the kl weight)

                kl_loss =  -0.5 * torch.sum(1 + log_var_z - mean_z ** 2 - log_var_z.exp(), dim=1)
                kl_loss = kl_loss.mean()
                rec_loss = torch.linalg.norm(batch.view(B,-1) - mean_x.view(B,-1), dim=1)
                rec_loss = rec_loss.mean()
                kl_weight = self.config.kl_weight
                loss = rec_loss + kl_weight * kl_loss
            else:
                z, _ = self.encode(batch)
                mean_x, _ = self.decode(z)
                rec_loss = torch.linalg.norm(batch.view(B,-1) - mean_x.view(B,-1), dim=1)
                rec_loss = rec_loss.mean()
                kl_loss = torch.tensor(0.)
                loss = rec_loss

            return loss, rec_loss.detach() , kl_loss.detach()

        def handle_batch(self, batch):
            if isinstance(batch, (list, tuple)):
                batch = batch[0]
            return batch 
        
        def training_step(self, batch, batch_idx):
            batch = self.handle_batch(batch)
            loss, rec_loss, kl_loss = self.compute_loss(batch)
            self.log('train_loss', loss)
            self.log('train_rec_loss', rec_loss)
            self.log('train_kl_loss', kl_loss)
            return loss
        
        def validation_step(self, batch, batch_idx):
            batch = self.handle_batch(batch)
            loss, rec_loss, kl_loss = self.compute_loss(batch)
            self.log('val_loss', loss)
            self.log('val_rec_loss', rec_loss)
            self.log('val_kl_loss', kl_loss)
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
    if dataset_tag == DATASET_CIFAR10:
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'cifar10')
        load = args.load
        assert args.data.channels == 3
        assert args.data.image_size == 32
        train_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        if args.data.random_flip:
            train_transform.insert(0, transforms.RandomHorizontalFlip())
        
        init_ds = torchvision.datasets.CIFAR10(root=root, train=True, transform=cmp(train_transform), download=True)

    # Downscaler dataset
    if dataset_tag == DATASET_DOWNSCALER_HIGH:
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

    # FINAL DATASET

    final_ds, mean_final, var_final = get_final_dataset(args, init_ds, device)
    return init_ds, final_ds, mean_final, var_final


def get_final_dataset(args, init_ds, device):
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


def get_valid_test_datasets(args):
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
        # data_tag = args.data.dataset
        root = os.path.join(data_dir, 'cifar10')
        load = args.load
        assert args.data.channels == 3
        assert args.data.image_size == 32
        test_transform = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        valid_ds = None
        test_ds = torchvision.datasets.CIFAR10(root=root, train=False, transform=transforms.Compose(test_transform), download=True)

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
