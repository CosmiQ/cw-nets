from os import walk
import os
import numpy as np
from sklearn.model_selection import train_test_split
from cw_nets.keras_tools.keras_geotiff import RasterDataGenerator
from cw_nets.keras_tools.unet_keras import unet
from tqdm import tqdm
from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
from keras.optimizers import SGD, Adam, Adagrad

import numpy as np
from keras.models import Sequential
from keras import utils




path_to_data = "/home/dlindenbaum/cw-tiler/cw-tiler/AOI_6_Atlanta/"
f = []
for (dirpath, dirnames, filenames) in walk(path_to_data):
    f.extend(filenames)
    break
#print(f)
image_list = [os.path.join(path_to_data, filename) for filename in filenames if "_image.tif" in filename]
label_list = [os.path.join(path_to_data, filename) for filename in filenames if "_label.tif" in filename]



print("Number of images: {}".format(len(image_list)))
print("Number of labels: {}".format(len(label_list)))
image_list = np.sort(image_list)
label_list = np.sort(label_list)
image_list_filter = []
label_list_filter = []

import rasterio
for image, label in tqdm(zip(image_list, label_list)):
    with rasterio.open(image) as src:
        data = src.read()
        if data.max()==0:
            #print("bad")
            pass
            
            
        else:
            image_list_filter.append(image)
            label_list_filter.append(label)
    

image_list_train, image_list_test, label_list_train, label_list_test = train_test_split(image_list_filter, label_list_filter, test_size=0.2, random_state=42)


n_channels =8
n_width = 800
n_height = 800
n_classes = 1
batch_size = 5
params={"dim": (n_width, n_height),
       "batch_size": batch_size,
        "n_classes":n_classes,
        "n_channels":n_channels,
        "shuffle":True,
        "max_value": -1, # Use Max Value of Chip or you can specify a max value to normalize by
        "img_norm": "divide" #Divide by max value and normalize to 0,1
       }


model = unet((n_channels, n_width, n_height), n_classes=1,
             kernel=3
            )

training_generator = RasterDataGenerator(image_list_train, label_list_train, **params)
validation_generator = RasterDataGenerator(image_list_test, label_list_test, **params)


 # set callbacks
print ("Setting callbacks...")
model_name = "AOI_6_unet_model_v1"
early_stopping_patience = 4

model_checkpoint = ModelCheckpoint(model_name, monitor='val_loss', 
                                           save_best_only=True)
early_stopping = EarlyStopping(monitor='val_loss', 
                                       patience=early_stopping_patience, 
                                       verbose=1, mode='auto')
tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True,                    write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

print ("Callbacks successfully set")


epochs=100



model.fit_generator(generator=training_generator,
                   validation_data=validation_generator,
                   use_multiprocessing=True,
                    workers=8,
                   verbose=1,
                   epochs=epochs,
                   callbacks=[model_checkpoint, early_stopping, tensorboard]
                   )
                   
