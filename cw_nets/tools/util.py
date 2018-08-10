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
import random
import torch
import json
import logging
import time
import io
from tqdm import tqdm

# Setting Certificate Location for Ubuntu/Mac OS locations (Rasterio looks for certs in centos locations)
os.environ['CURL_CA_BUNDLE']='/etc/ssl/certs/ca-certificates.crt'

logger = logging.getLogger(__name__)


def get_processing_details(rasterPath, smallExample=False, 
                             dstkwargs={"nodata": 0,
                                        "interleave": "pixel",
                                        "tiled": True,
                                        "blockxsize": 512,
                                        "blockysize": 512,
                                        "compress": "LZW"}):
    with rasterio.open(rasterPath) as src:

        # Get Lat, Lon bounds of the Raster (src)
        wgs_bounds = utils.get_wgs84_bounds(src)

        # Use Lat, Lon location of Image to get UTM Zone/ UTM projection
        utm_crs = utils.calculate_UTM_crs(wgs_bounds)

        # Calculate Raster bounds in UTM coordinates 
        utm_bounds = utils.get_utm_bounds(src, utm_crs)

        vrt_profile = utils.get_utm_vrt_profile(src,
                                               crs=utm_crs,
                                               )


        dst_profile = vrt_profile
        dst_profile.update({'count': 1,
                                'dtype': rasterio.uint8,
                            'driver': "GTiff",

                       })
        # update for CogStandard
        dst_profile.update(dstkwargs)


    # open s3 Location
    rasterBounds = geometry.box(*utm_bounds)

    if smallExample:
        rasterBounds = geometry.box(*rasterBounds.centroid.buffer(1000).bounds)
    
    return rasterBounds, dst_profile

def generate_cells_list_dict(rasterBounds, cell_size_meters, stride_size_meters, tile_size_pixels, quad_space=True):
    
    cells_list_dict = main.calculate_analysis_grid(rasterBounds.bounds, 
                                                   stride_size_meters=stride_size_meters, 
                                                   cell_size_meters=cell_size_meters,
                                                  quad_space=True)
    
    return cells_list_dict

def createRasterMask(rasterPath, 
                     cells_list_dict, 
                     dataLocation, 
                     outputName, 
                     dst_profile, 
                     modelPath,
                    tile_size_pixels,
                    logger=None):
    
    logger = logger or logging.getLogger(__name__)

    
    
    mask_dict_list = []
    model = tn_tools.get_model(modelPath)
    outputTifMask = os.path.join(dataLocation, outputName.replace('.tif', '_mask.tif'))
    outputTifCountour = os.path.join(dataLocation, outputName.replace('.tif', '_contour.tif'))
    outputTifCount = os.path.join(dataLocation, outputName.replace('.tif', '_count.tif'))


    # define Image_transform for Tile
    img_transform = tn_tools.get_img_transform()
    # Open Raster File
    with rasterio.open(rasterPath) as src:

        for cells_list_id, cells_list in cells_list_dict.items():

            outputTifMask = os.path.join(dataLocation, outputName.replace('.tif', '{}_mask.tif'.format(cells_list_id)))
            outputTifCountour = os.path.join(dataLocation, outputName.replace('.tif', '{}_contour.tif'.format(cells_list_id)))
            outputTifCount = os.path.join(dataLocation, outputName.replace('.tif', '{}_count.tif'.format(cells_list_id)))

            # Open Results TIF
            with rasterio.open(outputTifMask,
                                   'w',
                                   **dst_profile) as dst, \
                rasterio.open(outputTifCountour,
                                   'w',
                                   **dst_profile) as dst_countour, \
                rasterio.open(outputTifCount,
                                   'w',
                                   **dst_profile) as dst_count:

                src_profile = src.profile

                print("start interating through {} cells".format(len(cells_list_dict[0])))
                for cell_selection in tqdm(cells_list):
                    # Break up cell into four gorners
                    ll_x, ll_y, ur_x, ur_y = cell_selection

                    
                        # Get Tile from bounding box
                    tile, mask, window, window_transform = main.tile_utm(src, ll_x, ll_y, ur_x, ur_y, indexes=None, tilesize=tile_size_pixels, nodata=None, alpha=None,
                                     dst_crs=dst_profile['crs'])


                    img = tn_tools.reform_tile(tile)
                    img, pads = tn_tools.pad(img)

                    input_img = torch.unsqueeze(img_transform(img / 255).cuda(), dim=0)

                    predictDict = tn_tools.predict(model, input_img, pads)
                    # Returns predictDict = {'mask': mask, # Polygon Results for detection of buildings
                          # 'contour': contour, # Contour results for detecting edge of buildings
                          # 'seed': seed, # Mix of Contour and Mask for used by watershed function
                          # 'labels': labels # Result of watershed function
                        #} 


                    try: 
                        dst.write(tn_tools.unpad(predictDict['mask'], pads).astype(np.uint8), window=window, indexes=1)
                        dst_countour.write(tn_tools.unpad(predictDict['seed'], pads).astype(np.uint8), window=window, indexes=1)
                        dst_count.write(np.ones(predictDict['labels'].shape).astype(np.uint8), window=window, indexes=1)
                    
                    except (SystemExit, KeyboardInterrupt):
                        raise
                    
                    except Exception:
                        
                        logger.error("Failed To write tile:")
                        logger.error("Failed window: {}".format(window))
                        logger.error("Failed cell_section: {}".format(cell_selection))
                        
                        
                        
                        
                        
                        
                        
                        
            
            resultDict = {'mask': outputTifMask,
                         'contour': outputTifCountour,
                         'count': outputTifCount}
            
            
            mask_dict_list.append(resultDict)
            
            
            
            
    return mask_dict_list
            
def process_results_mask(mask_dict_list, outputNameTiff,  delete_tmp=True):
    firstCell = True
    src_mask_list     = []
    src_countour_list = []
    src_count_list    = []
    
    for resultDict in tqdm(mask_dict_list):
        
        
        src_mask_list.append(rasterio.open(resultDict['mask']))
        src_countour_list.append(rasterio.open(resultDict['contour'])) 
        src_count_list.append(rasterio.open(resultDict['count']))
        
            
    src_mask_profile = src_mask_list[0].profile
            
    with rasterio.open(outputNameTiff,
                                   'w',
                               **src_mask_profile) as dst:
                
        windows = [window for ij, window in dst.block_windows()]
                
        for window in tqdm(windows):
            firstCell = True
            
            for src_mask, src_contour, src_count in zip(src_mask_list, src_countour_list, src_count_list):
                
                if firstCell:
                    data_mask = src_mask.read(window=window)
                    data_count = src_count.read(window=window)
                    firstCell = False
                else:
                    data_mask += src_mask.read(window=window)
                    data_count += src_count.read(window=window)
            

                data_mask=(data_mask/data_count).astype(np.uint8)
                data_mask=(data_mask>=1.0).astype(np.uint8)
    
    
                dst.write(data_mask, window=window)
        
    
    resultDict = {'mask': outputNameTiff}
        
    
    return resultDict

def polygonize_results_mask(maskDict):
    
    results = []
    #mask= data_mask==0
    with rasterio.open(maskDict['mask']) as src:
        src_profile = src.profile
        image = src.read(1)
        mask=image>0
        for i, (geom, val) in tqdm(enumerate(rasterio.features.shapes(image, mask=mask, transform=src.transform))):
            geom = rasterio.warp.transform_geom(src.crs, 'EPSG:4326', geom, precision=6)
            results.append({
                "type": "Feature", 
                'properties': {'raster_val': val}, 
                'geometry': geom
            }
                          )
    
        
    return results, src_profile

def write_results_tojson(results, dst_name):
    
    

    collection = {
        'type': 'FeatureCollection', 
        'features': list(results) }

    with open(dst_name, 'w') as dst:
        json.dump(collection, dst)


