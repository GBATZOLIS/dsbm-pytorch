import math
import random

from PIL import Image
import blobfile as bf
#from mpi4py import MPI
import numpy as np
from torch.utils.data import DataLoader, Dataset

import multiprocessing
import torch
from torchvision import transforms, datasets
from tqdm import tqdm
import pytorch_lightning as pl
import os
from typing import List

def load_dataset_filenames(data_dir: str, train: bool, percentage_use=100, train_percentage: float = 0.8, val_percentage: float = 0.1) -> List[str]:
    """
    Load dataset filenames for training, validation, or testing.
    
    Args:
    - data_dir (str): Directory containing the dataset.
    - dataset_split (str): 'train', 'val', or 'test' to specify the dataset part.
    - train_percentage (float): Percentage of data to be used for training.
    - val_percentage (float): Percentage of data to be used for validation.

    Returns:
    - List[str]: List of image file paths.
    """
    # List all files
    all_files = _list_image_files_recursively(data_dir)
    
    # Fix the seed for reproducibility
    random.seed(42)
    all_files = random.sample(all_files, int(percentage_use/100*len(all_files)))
    random.shuffle(all_files)

    # Calculate dataset sizes
    total_images = len(all_files)
    train_size = int(total_images * train_percentage)
    val_size = int(total_images * val_percentage)

    if train:
        return all_files[:train_size]
    else:
        return all_files[train_size:train_size + val_size], all_files[train_size + val_size:]

def create_dataset(image_paths: List[str], random_crop: bool, image_size: int, random_flip: bool):
    """
    Create a PyTorch dataset from the list of image paths.

    Args:
    - image_paths (List[str]): List of image file paths.
    - random_crop (bool): Whether to apply random cropping.
    - image_size (int): Size of the images.
    - random_flip (bool): Whether to apply random flipping.

    Returns:
    - ImageDataset: The PyTorch dataset.
    """
    images = load_images(image_paths, random_crop, image_size)
    dataset = ImageDataset(
        resolution=image_size,
        images=images,
        random_crop=random_crop,
        random_flip=random_flip,
    )
    return dataset


def _list_image_files_recursively(data_dir):
    results = []
    for entry in sorted(bf.listdir(data_dir)):
        full_path = bf.join(data_dir, entry)
        ext = entry.split(".")[-1]
        if "." in entry and ext.lower() in ["jpg", "jpeg", "png", "gif"]:
            results.append(full_path)
        elif bf.isdir(full_path):
            results.extend(_list_image_files_recursively(full_path))
    return results

def load_image(params):
    path, random_crop, resolution = params
    with bf.BlobFile(path, "rb") as f:
        pil_image = Image.open(f)
        pil_image.load()
        pil_image = pil_image.convert("RGB")
        if random_crop:
            arr = random_crop_arr(pil_image, resolution)
        else:
            arr = center_crop_arr(pil_image, resolution)
        arr = arr.astype(np.float32) / 127.5 - 1  # Normalization
    return arr

def load_images(image_paths, random_crop, resolution):
    with multiprocessing.Pool() as pool:
        params = [(path, random_crop, resolution) for path in image_paths]
        images = list(tqdm(pool.imap(load_image, params), total=len(image_paths)))
    return images


class ImageDataset(Dataset):
    def __init__(
        self,
        resolution,
        images,
        classes=None,
        random_crop=False,
        random_flip=True,
    ):
        super().__init__()
        self.resolution = resolution
        self.local_images = images
        self.local_classes = classes
        self.random_crop = random_crop
        self.random_flip = random_flip

    def __len__(self):
        return len(self.local_images)

    def __getitem__(self, idx):
        arr = self.local_images[idx]

        if self.random_flip and random.random() < 0.5:
            arr = np.ascontiguousarray(arr[:, ::-1])

        out_dict = {}
        if self.local_classes is not None:
            out_dict["y"] = np.array(self.local_classes[idx], dtype=np.int64)
        return np.transpose(arr, [2, 0, 1]), out_dict


def center_crop_arr(pil_image, image_size):
    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]


def random_crop_arr(pil_image, image_size, min_crop_frac=0.8, max_crop_frac=1.0):
    min_smaller_dim_size = math.ceil(image_size / max_crop_frac)
    max_smaller_dim_size = math.ceil(image_size / min_crop_frac)
    smaller_dim_size = random.randrange(min_smaller_dim_size, max_smaller_dim_size + 1)

    # We are not on a new enough PIL to support the `reducing_gap`
    # argument, which uses BOX downsampling at powers of two first.
    # Thus, we do it by hand to improve downsample quality.
    while min(*pil_image.size) >= 2 * smaller_dim_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = smaller_dim_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = random.randrange(arr.shape[0] - image_size + 1)
    crop_x = random.randrange(arr.shape[1] - image_size + 1)
    return arr[crop_y : crop_y + image_size, crop_x : crop_x + image_size]