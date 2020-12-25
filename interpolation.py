import argparse
import os

import torch
import torch.nn as nn
import numpy as np

import utils.utils as utils

from torch import Tensor
from utils.constants import GANType
from torchvision.utils import save_image


def linear_interpolation(p0: Tensor, p1: Tensor, t: float) -> Tensor:
    r"""Interpolate between two images.

    Args:
        -p0 (Tensor): First endpoint.
        -p1 (Tensor): Second endpoint.
        -t (float): Interpolation parameter.

    Returns:
        -(Tensor): Interpolated image.
    """

    return p0 + t * (p1 - p0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path of pretrained GAN model.")
    parser.add_argument("--img_size", type=int, default=128,
                        help="Size of upsampled image.")
    parser.add_argument("--n_steps", type=int, default=16,
                        help="Number of interpolation steps.")
    parser.add_argument("--n_images", type=int, default=500,
                        help="How many interpolations to generate.")
    args = parser.parse_args()

    data_dir = "./data/interpolation"
    os.makedirs(data_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    G, _ = utils.get_gan(GANType.SN_DCGAN, device)
    G.load_state_dict(torch.load(args.ckpt_path))
    G.eval()

    for i in range(args.n_images):
        noise = utils.get_latent_batch(2, device)
        upsampler = nn.Upsample(size=(args.img_size, args.img_size),
                                mode="bilinear",
                                align_corners=True)

        imgs = torch.zeros(args.n_steps, 3, args.img_size, args.img_size)
        with torch.no_grad():
            for idx, t in enumerate(np.linspace(0, 1, args.n_steps)):
                inter_noise = linear_interpolation(noise[0], noise[1], t)
                inter_noise = torch.unsqueeze(inter_noise, dim=0)
                inter_img = G(inter_noise).cpu()
                inter_img = upsampler(inter_img)

                imgs[idx] = inter_img

        path = os.path.join(data_dir, f"inter{str(i)}.png")
        save_image(imgs, path, nrow=args.n_steps // 2, normalize=True)
