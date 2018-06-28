import numpy as np
import keras
import rasterio
from scipy.ndimage.interpolation import zoom

class RasterDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, batch_size=32, dim=(1200,1200), n_channels=8,
                 n_classes=2, shuffle=True, max_value=-1, img_norm="divide"):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.max_value=max_value
        self.img_norm=img_norm
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_image_temp = [self.list_IDs[k] for k in indexes]
        list_labels_temp = [self.labels[k] for k in indexes]


        # Generate data
        X, y = self.__data_generation(list_image_temp, list_labels_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp, list_labels_temp):
        'Generates data containing batch_size samples' # X : (n_samples, n_channels, *dim)
        # Initialization
        X = np.empty((self.batch_size, self.n_channels, *self.dim))
        y = np.empty((self.batch_size, self.n_classes, *self.dim))


        # Generate data
        for i, ID in enumerate(zip(list_IDs_temp, list_labels_temp)):
            # Store sample
            with rasterio.open(ID[0]) as src:
                data = src.read().astype(float)
                
                if self.max_value==-1:
                    self.max_value = np.max(data)
                
                if self.img_norm == "divide":    
                    X[i,] = np.clip(data*1.0/self.max_value, 0, 1)
                
                elif self.img_norm == "sub_and_divide":
                    X[i,] = np.clip((data*1.0/(self.max_value/2) - 1), -1, 1)

                
            with rasterio.open(ID[1]) as src:
                # Store class
                #data = src.read()[0]
                y[i] = src.read()
        
        

        return X, y