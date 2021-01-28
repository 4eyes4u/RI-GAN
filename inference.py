import argparse
import math
import os

import torch
import numpy as np

import utils.utils as utils

from utils.constants import GANType
from torch.utils.tensorboard import SummaryWriter
from models.dcgan import Generator, Discriminator
from torchvision.utils import save_image, make_grid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path of pretrained GAN model.")
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Size of batch for GAN inference.")
    parser.add_argument("--num_images", type=int, required=True,
                        help="How many images to generate.")
    args = parser.parse_args()

    data_dir = "./data/fake/fake"
    os.makedirs(data_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # since norm-layers are frozen in eval, we can use DCGAn for both confs
    G, _ = utils.get_gan(GANType.DCGAN, device, 3)
    G.load_state_dict(torch.load(args.ckpt_path))
    G.eval()

    img_cnt = 0
    with torch.no_grad():
        while img_cnt < args.num_images:
            noise = utils.get_latent_batch(args.batch_size, device)
            fake_imgs = G(noise).cpu()

            for idx, fake_img in enumerate(fake_imgs):
                path = os.path.join(data_dir, str(img_cnt + idx) + ".png")
                save_image(fake_img, path, normalize=True)

            img_cnt += args.batch_size
