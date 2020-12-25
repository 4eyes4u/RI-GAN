# Training
One should not change anything in `train.py` prior to training. For that purpose, adjust all parameters in `train_config.json`.

* `num_epochs` - number of epochs for training.
* `batch_size` - size of batch during training.
* `img_size` - size of images for both discriminator and generator.
* `checkpoint_freq` - at how many epochs to dump GAN state.
* `log_freq` - at how many batches per epoch to dump log.
* `imagery_freq` - at how many batches per poech to save intermediate results.
* `n_power_iterations` - number of iterations used for calculating l<sub>2</sub> norm of weight matrix.
* `type` - kind of GAN architecture. Either `DCGAN` or `SN_DCGAN`.

CelebA will be downloaded in folder `./data` in case it's not already. Log and images will be saved in proper `./runs` subdirectory.

# Testing
* `is.py` - calculating Inception score. See comments in the file for more info.
* `interpolation.py` - interpolate between given images proper number of times. Saves images to `./data/interpolation`.
* `inference.py`- generated given number of fake images for provided pre-trained model.

# Checkpoints
All pretrained models are in `./runs` directory. One can check training parameters in proper `train_config.json` files.