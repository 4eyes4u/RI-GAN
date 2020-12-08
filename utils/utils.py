import os
import torch
import torch.nn as nn
import zipfile
import logging
import shutil

from .constants import DATA_DIR, LATENT_SPACE_DIM

from torch import Tensor
from torch.optim import Optimizer
from torch.hub import download_url_to_file
from torch.utils.data import DataLoader
from torch.optim import Adam

from torchvision import transforms, datasets
from torchvision.datasets import ImageFolder

from typing import Tuple

from models.dcgan import Generator, Discriminator


def prepare_celeba(celeba_path: str, img_size: int):
    r"""Downloading and preparing CelebA dataset.

    Args:
        -celeba_path (str): Where to store dataset.
        -img_size (int): Size of singleton image in the dataset.
    """

    celeba_url = r"https://s3.amazonaws.com/video.udacity-data.com/topher/2018/November/5be7eb6f_processed-celeba-small/processed-celeba-small.zip"

    # download dataset
    print("downloading CelebA...")
    resource_tmp_path = celeba_path + ".zip"
    download_url_to_file(celeba_url, resource_tmp_path)
    print("finished")

    # unzipping downloaded dataset
    print("unzipping...")
    with zipfile.ZipFile(resource_tmp_path) as zf:
        os.makedirs(celeba_path, exist_ok=True)
        zf.extractall(path=celeba_path)
    print("finished")

    # removing temporary files
    os.remove(resource_tmp_path)
    print("removed temporary file")

    # preparing dataset
    print("preparing CelebA...")
    shutil.rmtree(os.path.join(celeba_path, "__MACOSX"))
    dst_data_directory = os.path.join(celeba_path, "processed_celeba_small")
    os.remove(os.path.join(dst_data_directory, ".DS_Store"))
    data_directory1 = os.path.join(dst_data_directory, "celeba")
    data_directory2 = os.path.join(data_directory1, "New Folder With Items")
    for element in os.listdir(data_directory1):
        if not element.endswith(".jpg"):
            continue

        if os.path.isfile(os.path.join(data_directory1, element)):
            shutil.move(os.path.join(data_directory1, element),
                        os.path.join(dst_data_directory, element))

    for element in os.listdir(data_directory2):
        if not element.endswith(".jpg"):
            continue

        if os.path.isfile(os.path.join(data_directory2, element)):
            shutil.move(os.path.join(data_directory2, element),
                        os.path.join(dst_data_directory, element))

    shutil.rmtree(data_directory1)
    print("finished")


def get_data_loader(batch_size: int, img_size: int) -> DataLoader:
    r"""Create `DataLoader` for CelebA dataset.

    Args:
        -batch_size (int): Size of batch.
        -img_size (int): Size of image.

    Returns:
        -celeba_data_loader (DataLoader): `DataLoader`.
    """

    def get_transform():
        transform = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5), (0.5))
        ])

        return transform

    # checking if dataset has already been downloaded
    celeba_path = os.path.join(DATA_DIR, "CelebA")
    if not os.path.exists(celeba_path):
        prepare_celeba(celeba_path, img_size)

    celeba_dataset = ImageFolder(celeba_path,
                                 transform=get_transform())
    celeba_data_loader = DataLoader(celeba_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    drop_last=True)

    return celeba_data_loader


def get_gan(device: torch.device) -> Tuple[nn.Module, nn.Module]:
    r"""Fetching GAN and moving it to proper device.

    Args:
        -device (torch.device): On which device (eg. GPU) to move models.

    Returns:
        -G (nn.Module): Generator.
        -D (nn.Module): Discriminator.
    """

    G = Generator().to(device)
    D = Discriminator().to(device)

    return G, D


def get_optimizers(G: nn.Module,
                   D: nn.Module) -> Tuple[Optimizer, Optimizer]:
    r"""Initializing optimizers for GAN.

    Args:
        -G (nn.Module): Generator.
        -D (nn.Module): Discriminator.

    Returns:
        -G_opt (nn.Optimizer): Generator's Adam optimizer.
        -D_opt (nn.Optimizer): Discriminator's Adam optimizer.
    """

    G_optim = Adam(G.parameters(), lr=0.002, betas=(0.5, 0.999))
    D_optim = Adam(D.parameters(), lr=0.002, betas=(0.5, 0.999))

    return G_optim, D_optim


def get_gt(batch_size: int, device: torch.device) -> Tuple[Tensor, Tensor]:
    r"""Initializing ground truth labels for both real and fake images.

    Args:
        -batch_size (int): Batch size.
        -device (torch.device): On which device (eg. GPU) to move models.

    Returns:
        -real_gt (Tensor): Ground truth for real images.
        -fake_gt (Tensor): Ground truth for fake images.
    """

    real_gt = torch.ones(batch_size, 1, 1, 1).to(device)
    fake_gt = torch.zeros(batch_size, 1, 1, 1).to(device)

    return real_gt, fake_gt


def get_latent_batch(batch_size: int, device: torch.device) -> Tensor:
    r"""Create latent batch of Gaussian noise.

    Args:
        -batch_size (int): Batch size.
        -device (torch.device): On which device (eg. GPU) to move models.

    Returns:
        -latent_batch (Tensor): Noise tensor of provided size.
    """

    latent_batch = torch.randn(batch_size, LATENT_SPACE_DIM).to(device)

    return latent_batch
