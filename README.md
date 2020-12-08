# Training
One should not change anything in `train.py` prior to training. For that purpose, adjust all parameters in `train_config.json`.

* `num_epochs` - number of epochs for training
* `batch_size` - size of batch during training
* `img_size` - size of images for both discriminator and generator
* `checkpoint_freq` - at how many epochs to dump GAN state
* `log_freq` - at how many batches per epoch to dump log