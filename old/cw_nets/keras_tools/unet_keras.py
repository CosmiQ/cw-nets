## Code modified from https://github.com/CosmiQ/basiss/blob/master/src/basiss.py
from keras.models import Model
from keras.layers import (Dense, Dropout, Activation, Flatten, Reshape, 
                          Lambda, Convolution2D, Conv2D, MaxPooling2D, 
                          UpSampling2D, Input, merge, Concatenate, 
                          concatenate, Conv2DTranspose)
from keras.callbacks import ModelCheckpoint, EarlyStopping
                        #,LearningRateScheduler
from keras.optimizers import SGD, Adam, Adagrad
import keras.utils.vis_utils
from cw_nets.keras_tools.keras_callbacks import jaccard_coef, jaccard_coef_int, dice_coeff, f1_score

###############################################################################
### Define model(s)
###############################################################################
def unet(input_shape, n_classes=2, kernel=3, loss='binary_crossentropy', 
         optimizer='SGD', data_format='channels_first', multi_gpu=1):
    '''
    https://arxiv.org/abs/1505.04597
    https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py
    # modified from UNET to handle channels_first data explicitly from 
    '''
    
    print ("UNET input shape:", input_shape) 
    #inputs = Input((input_size, input_size, n_channels))
    inputs = Input(input_shape)
    
    conv1 = Conv2D(32, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(inputs)
    conv1 = Conv2D(32, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv1)

    conv2 = Conv2D(64, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(pool1)
    conv2 = Conv2D(64, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv2)

    conv3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(pool2)
    conv3 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv3)

    conv4 = Conv2D(256, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(pool3)
    conv4 = Conv2D(256, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), data_format=data_format)(conv4)

    conv5 = Conv2D(512, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(pool4)
    conv5 = Conv2D(512, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(conv5)

    up6 = concatenate([Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same', data_format=data_format)(conv5), conv4], axis=1)
    #up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Conv2D(256, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(up6)
    conv6 = Conv2D(256, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same', data_format=data_format)(conv6), conv3], axis=1)
    #up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(up7)
    conv7 = Conv2D(128, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same', data_format=data_format)(conv7), conv2], axis=1)
    #up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Conv2D(64, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(up8)
    conv8 = Conv2D(64, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same', data_format=data_format)(conv8), conv1], axis=1)
    #up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Conv2D(32, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(up9)
    conv9 = Conv2D(32, (kernel, kernel), activation='relu', padding='same', data_format=data_format)(conv9)

    conv10 = Conv2D(n_classes, (1, 1), activation='sigmoid', data_format=data_format)(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    
    if optimizer.upper() == 'SGD':
        opt_f = SGD()
    elif optimizer.upper == 'ADAM':
        opt_f = Adam()
    elif optimizer.upper == 'ADAGRAD':
        opt_f = Adagrad()
    else:
        print ("Unknown optimzer:", optimizer)
        return
    
    
    
    model.compile(optimizer=opt_f, loss=loss, 
                  metrics=[jaccard_coef, jaccard_coef_int, dice_coeff, 
                           'accuracy', 'mean_squared_error', f1_score])#,
                           #'precision', 'recall', 'f1score', 'mse'])

    print ("UNET Total number of params:", model.count_params() )     
    return model