import os
import logging
import argparse
from cw_tiler import main
from cw_tiler import utils
from cw_tiler import vector_utils
from cw_nets.Ternaus_tools import tn_tools 
from cw_nets.tools import util as base_tools


if __name__ == '__main__':
    
    logger = logging.getLogger(__name__)

    
    parser = argparse.ArgumentParser(description='Perform Building Extraction on Imagery')
    parser.add_argument("--raster_path",
                       help="Location to GeoTiff for processing")
    parser.add_argument("--output_name",
                       help="Output Base Name.tif")
    parser.add_argument("--data_output",
                       help="Location to Save Data too")
    parser.add_argument("--model_path", 
                       help="Location of Model")
    parser.add_argument("--cell_size", type=int,
                       help="Cell Size of sliding window in Meters")
    parser.add_argument("--stride_size", type=int,
                       help="Stride of Sliding window in meters")
    parser.add_argument("--tile_size", type=int,
                       help="number of pixels in a tile cell_size_meters/tile_size_pixels")
    parser.add_argument('--debug', action='store_true', help='print debug messages to stdout')
    parser.add_argument('--sample', action='store_true', help='Process Sample of Raster')
    parser.add_argument('--quad_space', action='store_true', help='print debug messages to stderr')




    args = parser.parse_args()

    ## build logger
    # Create the Logger
    logger = logging.getLogger(__name__)
    if args.debug:
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
    
    if args.debug:
        logger_handler.setLevel(logging.DEBUG)
    else:
        logger_handler.setLevel(logging.INFO)
    
    logger.addHandler(logger_handler)
        
        
    
    logger.info("Starting Process")
    rasterBounds, dst_profile = base_tools.get_processing_details(args.raster_path, smallExample=args.sample)
    cells_list_dict = base_tools.generate_cells_list_dict(rasterBounds, args.cell_size, args.stride_size, args.tile_size, quad_space=args.quad_space)
    
    logger.info("Starting inference Process")
    mask_dict_list = base_tools.createRasterMask(args.raster_path, cells_list_dict, args.data_output, 
                                                 args.output_name, dst_profile, args.model_path, args.tile_size)
    logger.info("Starting Combination Process")
    resultDict = base_tools.process_results_mask(mask_dict_list, 
                                            os.path.join(args.data_output, args.output_name), 
                                           )
    
    logger.info("Starting Polgonization Process")
    results, src_profile = base_tools.polygonize_results_mask(resultDict)

    logger.info("Writing Results to JSON")
    base_tools.write_results_tojson(results, 
                                os.path.join(args.data_output, args.output_name.replace('.tif', '.geojson')))
    
    logger.info("Finished")
             
        
        
    

    

        
    
    
    
    
    
    
 
    
    
    




