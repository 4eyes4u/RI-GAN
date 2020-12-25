import os

from enum import Enum


# Enum class for GAN type
class GANType(Enum):
    DCGAN = 1
    SN_DCGAN = 2


# dimension of latent space used for noise generation
LATENT_SPACE_DIM = 100

# directory where data is stored
DATA_DIR = os.path.join(os.path.dirname(__file__), os.pardir, "data")
os.makedirs(DATA_DIR, exist_ok=True)
