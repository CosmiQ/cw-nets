## Notebook for Creating Generator code for Keras and PyTorch
from __future__ import print_function, division
import os
import torch
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
#
# Import base tools

## Note, for mac osx compatability import something from shapely.geometry before importing fiona or geopandas
## https://github.com/Toblerity/Shapely/issues/553  * Import shapely before rasterio or fioana
from shapely import geometry
import rasterio
import random
from cw_tiler import main
from cw_tiler import utils
from cw_tiler import vector_utils
import numpy as np
import os
from tqdm import tqdm
import random
import cv2
import logging
# Setting Certificate Location for Ubuntu/Mac OS locations (Rasterio looks for certs in centos locations)
## TODO implement os check before setting
os.environ['CURL_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'
from cw_nets.tools import util as base_tools


argsdebug=True

logger = logging.getLogger(__name__)
if argsdebug:
    logger.setLevel(logging.DEBUG)
else:
    logger.setLevel(logging.INFO)

# Create the Handler for logging data to a file
logger_handler = logging.StreamHandler()
# Create a Formatter for formatting the log messages
logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

# Add the Formatter to the Handler
logger_handler.setFormatter(logger_formatter)

# Add the Handler to the Logger
    
if argsdebug:
    logger_handler.setLevel(logging.DEBUG)
else:
    logger_handler.setLevel(logging.INFO)
logger.addHandler(logger_handler)


class largeGeoTiff(Dataset):
    """Face Landmarks dataset."""
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    def __init__(self, raster_path, 
                 stride_size_meters=150,
                 cell_size_meters = 200,
                 tile_size_pixels = 650, 
                 transform=None,
                 quad_space=False,
                 sample=False,
                 testing=True
                ):
        """
        Args:
            rasterPath (string): Path to the rasterFile
            stride_size_meters (float): sliding window stride size in meters
            cell_size_meters (float): sliding window size in meters
            tile_size_pixels (float): sliding window pixel dimensions
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        # Create the Handler for logging data to a file
        logger_handler = logging.StreamHandler()
        # Create a Formatter for formatting the log messages
        logger_formatter = logging.Formatter('%(name)s - %(levelname)s - %(message)s')

        # Add the Formatter to the Handler
        logger_handler.setFormatter(logger_formatter)

        # Add the Handler to the Logger

        if argsdebug:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)
    
        self.logger.addHandler(logger_handler)
        
        self.testing=testing

        self.raster_path = raster_path
        self.stride_size_meters = stride_size_meters
        self.cell_size_meters = cell_size_meters
        self.tile_size_pixels = tile_size_pixels
        self.transform = transform
        
        
        rasterBounds, dst_profile = base_tools.get_processing_details(self.raster_path, smallExample=sample)

        self.src = rasterio.open(self.raster_path)

        # Get Lat, Lon bounds of the Raster (src)
        self.wgs_bounds = utils.get_wgs84_bounds(self.src)

        # Use Lat, Lon location of Image to get UTM Zone/ UTM projection
        self.utm_crs = utils.calculate_UTM_crs(self.wgs_bounds)

        # Calculate Raster bounds in UTM coordinates 
        self.utm_bounds = utils.get_utm_bounds(self.src, self.utm_crs)
        
        self.rasterBounds = rasterBounds
        self.cells_list = base_tools.generate_cells_list_dict(rasterBounds, 
                                                              self.cell_size_meters, 
                                                              self.stride_size_meters, 
                                                              self.tile_size_pixels, 
                                                              quad_space=quad_space
                                                             )
        self.cells_list = self.cells_list[0]


        if self.testing:
            with rasterio.open("test.tif", "w", **dst_profile) as dst:
                self.cells_list = [window for ij, window in dst.block_windows()]
        
            
        
        



            

    def __len__(self):
        return len(self.cells_list)

    def __getitem__(self, idx):
        
        # Get Tile from bounding box
        source_Raster=False
        if source_Raster:
            src_ras = self.raster_path
        else:
            src_ras  = self.src
        
        
        if self.testing:
            sample = src_ras.read(window=self.cells_list[idx])
        else:
            cell_selection = self.cells_list[idx]
            ll_x, ll_y, ur_x, ur_y = cell_selection


            tile, mask, window, window_transform = main.tile_utm(src_ras, 
                                                             ll_x, ll_y, ur_x, ur_y, 
                                                             indexes=None, 
                                                             tilesize=self.tile_size_pixels, 
                                                             nodata=None, 
                                                             alpha=None,
                                                             dst_crs=self.utm_crs)
        #except:
        #    print(cell_selection)
        
        
            sample = {'tile': tile.astype(np.float), 
                  'mask': mask,
                 'window': window.toranges(),
                 'window_transform': window_transform}

            if self.transform:
                sample = self.transform(sample)

        return sample
    
from pylab import *
from skimage.morphology import watershed
import scipy.ndimage as ndimage
from PIL import Image, ImagePalette

from torch.nn import functional as F
from torchvision.transforms import ToTensor, Normalize, Compose
import torch

import tifffile as tiff
import cv2
import random
from pathlib import Path

img_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406, 0, 0, 0, 0, 0, 0, 0, 0], 
              std=[0.229, 0.224, 0.225, 1, 1, 1, 1, 1, 1, 1, 1])
])

def pad(img, pad_size=32):
    """
    Load image from a given path and pad it on the sides, so that eash side is divisible by 32 (network requirement)
    if pad = True:
        returns image as numpy.array, tuple with padding in pixels as(x_min_pad, y_min_pad, x_max_pad, y_max_pad)
    else:
        returns image as numpy.array
    """

    if pad_size == 0:
        return img

    height, width = img.shape[:2]

    if height % pad_size == 0:
        y_min_pad = 0
        y_max_pad = 0
    else:
        y_pad = pad_size - height % pad_size
        y_min_pad = int(y_pad / 2)
        y_max_pad = y_pad - y_min_pad

    if width % pad_size == 0:
        x_min_pad = 0
        x_max_pad = 0
    else:
        x_pad = pad_size - width % pad_size
        x_min_pad = int(x_pad / 2)
        x_max_pad = x_pad - x_min_pad

    img = cv2.copyMakeBorder(img, y_min_pad, y_max_pad, x_min_pad, x_max_pad, cv2.BORDER_REFLECT_101)

    return img, (x_min_pad, y_min_pad, x_max_pad, y_max_pad)

def minmax(img):
    out = np.zeros_like(img).astype(np.float32)
    if img.sum() == 0:
        return out

    for i in range(img.shape[2]):
        c = img[:, :, i].min()
        d = img[:, :, i].max()

        t = (img[:, :, i] - c) / (d - c)
        out[:, :, i] = t
    return out.astype(np.float32)

def reform_tile(tile, rollaxis=True):
    
    if rollaxis:
        tile = np.rollaxis(tile, 0,3)
    rgb = minmax(tile[:,:,(5,3,2)]) 
    
    tf = tile.astype(np.float32)/ (2**11 - 1)
    
    return np.concatenate([rgb, tf], axis=2) * (2**8 - 1)

def teranaus_transform(sample):
    """sample = {'tile': tile, 
                  'mask': mask,
                 'window': window,
                 'window_transform': window_transform}
                 
                 """
    
    img = reform_tile(sample['tile'])
    img, pads = pad(img)
    input_img = torch.unsqueeze(img_transform(img / 255), dim=0)
    
    sample.update({'pad_img': img,
                  'pads': pads})
    
    return sample


if __name__ == '__main__':
    stride_size_meter = 150
    cell_size_meter = 200
    tile_size_pixels = 650
    rasterPath = "/home/dlindenbaum/057341085010_01_assembley_MULPan_cog.tif"
    rasterPath = "/nfs/data/Datasets/CosmiQ_SpaceNet_Src/AOI_2_Vegas/srcData/rasterData/AOI_2_Vegas_MUL-PanSharpen_Cloud.tif"
    rasterPath = "s3://spacenet-dataset/AOI_2_Vegas/srcData/rasterData/AOI_2_Vegas_MUL-PanSharpen_Cloud.tif"


    spaceNetDatset = largeGeoTiff(rasterPath, 
                     stride_size_meters=stride_size_meter,
                     cell_size_meters = cell_size_meter,
                     tile_size_pixels = tile_size_pixels, 
                     transform=teranaus_transform,
                                  sample=True
                                 )
    dataloader = DataLoader(spaceNetDatset, batch_size=10,
                            shuffle=False, num_workers=2)

    from tqdm import tqdm

    for idx, sample in tqdm(enumerate(dataloader)):
        logger.info("Testing idx")
        if idx == 10:
            break
    