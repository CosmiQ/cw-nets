import os
from ..utils import config, im_exts
from train_utils import get_loss_function, get_callbacks, get_metrics
from ..data.datagen import DataGenerator

def train(config_path):
    """Train a model of interest."""

    model_config = config.parse(config_path)
    # TODO: IMPLEMENT TESTING OF IMAGERY, MASKS, ETC IN THEIR PATHS
    if config['pretrained']:
        # TODO: IMPLEMENT DOWNLOADING PRE-TRAINED MODEL WEIGHTS
        pass
    train_fnames = [f for f in
                    [os.listdir(d) for d in config['train_im_src_dirs']]
                    if os.path.splitext(f)[1].lower() in im_exts]
    train_masks = [f for f in
                   [os.listdir(d) for d in config['train_label_src_dirs']]
                   if os.path.splitext(f)[1].lower() in im_exts]
    val_fnames = [f for f in
                  [os.listdir(d) for d in config['val_im_src_dirs']]
                  if os.path.splitext(f)[1].lower() in im_exts]
    val_masks = [f for f in
                 [os.listdir(d) for d in config['val_label_src_dirs']]
                 if os.path.splitext(f)[1].lower() in im_exts]

    train_dg = DataGenerator(model_config['framework'], train_fnames,
                             train_masks, model_config)
    val_dg = DataGenerator(model_config['framework'], val_fnames,
                           val_masks, model_config)
    loss_function = get_loss_function(model_config)
    callbacks = get_callbacks(model_config['framework'],
                              model_config['train']['callbacks'])
    metrics = get_metrics(model_config)

    if model_config['framework'] == 'keras':
        train_keras(model_config, train_dg, val_dg)


def train_keras(model_config, train_datagen, val_datagen):
    """Train a keras implementation of a model."""
    # load in pretrained model
    if model_config['pretrained']:
        # TODO: load model from pretrained path
        pass
    # set up callbacks

    # set up metrics
    # set up loss function
