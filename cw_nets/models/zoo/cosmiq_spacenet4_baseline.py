"""The CosmiQ SpaceNet Challenge Round 4 Baseline model.

This code is a modified version of the model implementation in the
CosmiQ_SN4_Baseline repository: https://github.com/cosmiq/cosmiq_sn4_baseline

This model is a Keras implementation of an untrained version of TernausNet V1.
TernausNetV1 is a U-Net with a VGG11 encoder pretrained on ImageNet. For more,
see https://arxiv.org/abs/1801.05746 and https://github.com/ternaus/TernausNet.

"""
# random seed instantiated here for reproducibility
RANDOM_SEED = 1337
import numpy as np
np.random.seed(RANDOM_SEED)
import tensorflow as tf
tf.set_random_seed(RANDOM_SEED)
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose
from keras.layers import concatenate
from keras.models import Model
from keras.optimizers import SGD, Adam, Adagrad, Nadam


def cosmiq_spacenet4_baseline(input_shape=(512, 512, 3), base_depth=64,
                              lr=0.0001, optimizer='Nadam',
                              loss_func='binary_crossentropy',
                              additional_metrics=None, verbose=False):
    """Compile a Keras model for training.

    Arguments:
    ----------
    input_shape (3-tuple): a tuple defining the shape of the input image.
    base_depth (int): the base convolution filter depth for the first layer
        of the model. Must be divisible by two, as the final layer uses
        base_depth/2 filters. The default value, 64, corresponds to the
        original TernausNetV1 depth.
    lr (float): learning rate.
    optimizer (['Adam', 'Adagrad', 'Nadam', 'SGD', or optimizer instance]):
        Optimizer to use to train the model. If a string from the options
        above is passed, the Keras optimizer with the same name is called
        with the default arguments (except learning rate, which uses the
        value passed in `lr`.) Alternatively, users may instantiate a Keras
        optimizer themselves with the desired configuration arguments and pass
        it here. Defaults to Adam.
    loss_func (str or function): Loss function to use during training.
        As with most Keras model this can be a string (e.g. the default,
        "binary_crossentropy") or a function.
    additional_metrics (list of functions or strs): Metrics functions or strs
        compatible with Keras. These are added to ['acc', 'mean_squared_error']
        which are included by default.

    Returns:
    --------
    A compiled Keras model ready to use for training.

    """

    model = spacebase4(input_shape=input_shape, base_depth=base_depth)
    if optimizer == 'Adam':
        opt_f = Adam(lr=lr)
    elif optimizer == 'SGD':
        opt_f = SGD(lr=lr)
    elif optimizer == 'Adagrad':
        opt_f = Adagrad(lr=lr)
    elif optimizer == 'Nadam':
        opt_f = Nadam(lr=lr)
    else:
        opt_f = optimizer
    if additional_metrics is None:
        additional_metrics = []
    model.compile(optimizer=opt_f,
                  loss=loss_func,
                  metrics=['acc', 'mean_squared_error'] + additional_metrics)
    # model.summary()
    return model


def spacebase4(input_shape=(512, 512, 3), base_depth=64):
    """Keras implementation of untrained TernausNet model architecture.

    Arguments:
    ----------
    input_shape (3-tuple): a tuple defining the shape of the input image.
    base_depth (int): the base convolution filter depth for the first layer
        of the model. Must be divisible by two, as the final layer uses
        base_depth/2 filters. The default value, 64, corresponds to the
        original TernausNetV1 depth.

    Returns:
    --------
    An uncompiled Keras Model instance with TernausNetV1 architecture.

    """
    inputs = Input(input_shape)
    conv1 = Conv2D(base_depth, 3, activation='relu', padding='same')(inputs)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2_1 = Conv2D(base_depth*2, 3, activation='relu',
                     padding='same')(pool1)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2_1)

    conv3_1 = Conv2D(base_depth*4, 3, activation='relu',
                     padding='same')(pool2)
    conv3_2 = Conv2D(base_depth*4, 3, activation='relu',
                     padding='same')(conv3_1)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3_2)

    conv4_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(pool3)
    conv4_2 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(conv4_1)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4_2)

    conv5_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(pool4)
    conv5_2 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(conv5_1)
    pool5 = MaxPooling2D(pool_size=(2, 2))(conv5_2)

    conv6_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(pool5)

    up7 = Conv2DTranspose(base_depth*4, 2, strides=(2, 2), activation='relu',
                          padding='same')(conv6_1)
    concat7 = concatenate([up7, conv5_2])
    conv7_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(concat7)

    up8 = Conv2DTranspose(base_depth*4, 2, strides=(2, 2), activation='relu',
                          padding='same')(conv7_1)
    concat8 = concatenate([up8, conv4_2])
    conv8_1 = Conv2D(base_depth*8, 3, activation='relu',
                     padding='same')(concat8)

    up9 = Conv2DTranspose(base_depth*2, 2, strides=(2, 2), activation='relu',
                          padding='same')(conv8_1)
    concat9 = concatenate([up9, conv3_2])
    conv9_1 = Conv2D(base_depth*4, 3, activation='relu',
                     padding='same')(concat9)

    up10 = Conv2DTranspose(base_depth, 2, strides=(2, 2), activation='relu',
                           padding='same')(conv9_1)
    concat10 = concatenate([up10, conv2_1])
    conv10_1 = Conv2D(base_depth*2, 3, activation='relu',
                      padding='same')(concat10)

    up11 = Conv2DTranspose(int(base_depth/2), 2, strides=(2, 2),
                           activation='relu', padding='same')(conv10_1)
    concat11 = concatenate([up11, conv1])

    out = Conv2D(1, 1, activation='sigmoid', padding='same')(concat11)

    return Model(input=inputs, output=out)
