# Import base tools
import os
## Note, for mac osx compatability import something from shapely.geometry before importing fiona or geopandas
## https://github.com/Toblerity/Shapely/issues/553  * Import shapely before rasterio or fioana
from shapely import geometry
import rasterio
import random
from cw_tiler import main
from cw_tiler import utils
from cw_tiler import vector_utils
from cw_nets.Ternaus_tools import tn_tools 
import numpy as np
import os
from tqdm import tqdm
import random
import torch
# Setting Certificate Location for Ubuntu/Mac OS locations (Rasterio looks for certs in centos locations)
os.environ['CURL_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'


