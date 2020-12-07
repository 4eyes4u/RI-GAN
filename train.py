import torch
import json
import time
import os
import shutil
import logging

from argparse import Namespace

import torch.nn as nn
import utils.utils as utils

from torch.utils.tensorboard import SummaryWriter
from models.dcgan import Generator, Discriminator


def make_dir_hierarchy():
    r"""Creating all neccessary directories that current run will use."""

    timestamp = time.strftime("%y-%m-%d-%H-%M-%S", time.gmtime())

    # directory for storing information of current run
    runs_path = os.path.join("runs", timestamp)
    os.makedirs(runs_path, exist_ok=True)

    # directory for storing intermediate generated images
    imagery_path = os.path.join(runs_path, "imagery")
    os.makedirs(imagery_path, exist_ok=True)

    # directory for storing log (including loss information)
    log_path = os.path.join(runs_path, "log")
    os.makedirs(log_path, exist_ok=True)

    # directory for storing checkpoints
    checkpoints_path = os.path.join(runs_path, "checkpoints")
    os.makedirs(checkpoints_path, exist_ok=True)

    # copying configuration file
    shutil.copy2("train_config.json", runs_path)

    paths = {
        "runs_path": runs_path,
        "imagery_path": imagery_path,
        "log_path": log_path,
        "checkpoints_path": checkpoints_path
    }
    paths = Namespace(**paths)

    return paths


if __name__ == "__main__":
    # configurating logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s: [%(levelname)s] %(message)s",
        datefmt="%y-%m-%d %H:%M:%S"
    )

    # fetching device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.debug(f"{device}")

    # training configuration
    with open("train_config.json", "r") as f:
        train_config = json.load(f)
        args = Namespace(**train_config)

    paths = make_dir_hierarchy()

    # initializing networks and optimizers
    G, D = utils.get_gan(device)
    G_optim, D_optim = utils.get_optimizers(G, D)

    # initializing loader for data
    data_loader = utils.get_data_loader(args.batch_size, args.img_size)

    # setting up loss and GT
    adversarial_loss = nn.BCELoss()
    real_gt, fake_gt = utils.get_gt(args.batch_size, device)

    # for logging
    log_batch_size = 25
    log_noise = utils.get_latent_batch(log_batch_size, device)
    D_loss_values = []
    G_loss_values = []
    img_count = 0

    writer = SummaryWriter(paths.log_path)

    for epoch in range(args.num_epochs):
        for batch_idx, (real_batch, _) in enumerate(data_loader):
            real_batch = real_batch.to(device)

            # discriminator part
            D_optim.zero_grad()

            D_real_loss = adversarial_loss(D(real_batch), real_gt)

            latent_batch = utils.get_latent_batch(args.batch_size, device)
            fake_batch = G(latent_batch)
            D_fake_loss = adversarial_loss(D(fake_batch.detach()), fake_gt)

            D_loss = D_real_loss + D_fake_loss
            D_loss_values.append(D_loss.item())
            D_loss.backward()
            D_optim.step()

            # generator part
            G_optim.zero_grad()

            latent_batch = utils.get_latent_batch(args.batch_size, device)
            fake_batch = G(latent_batch)
            G_loss = adversarial_loss(D(fake_batch), real_gt)
            G_loss_values.append(G_loss.item())
            G_loss.backward()
            G_optim.step()

            # logging current checkpoint to TensorBoard
            tag_scalar_dict = {"G": G_loss.item(), "D": D_loss.item()}
            global_step = len(data_loader) * epoch + batch_idx + 1
            writer.add_scalars("loss", tag_scalar_dict, global_step)

            if batch_idx % args.log_freq == 0:
                fmt = [epoch, batch_idx + 1, len(data_loader)]
                logging.info("epoch={:>2} batch=[{}/{}]".format(*fmt))

        # dumping generator
        if (epoch + 1) % args.checkpoint_freq == 0:
            torch.save(G.state_dict(),
                       os.path.join(paths.checkpoints_path,
                                    f"dcgan_ckpt_epoch_{epoch + 1}.pth"))
