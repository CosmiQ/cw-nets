## This class is a template for creating a transferable model
import rasterio


class base_net():
    
    def __init__(self, GSD_meters, image_size, bands=[]):
        #Band Order assumed to be RGB is unspecified
        
        self.GSD_meters = gsd_meters
        self.image_size = image_size
        
        if not bands:
            self.bands = [1,2,3]
        else:
            self.bands = bands
            
    
    def load_img(file_path):
        
        with rasterio.open(file_path) as src:
            
        