=========
cw-nets
=========


Segmentation Nets designed for use with SpaceNet datasets and other remote sensing data

Example
------------
python create_mask.py --raster_path s3://spacenet-dataset/AOI_2_Vegas/srcData/rasterData/AOI_2_Vegas_MUL-PanSharpen_Cloud.tif \
        --output_name AOI_2_Vegas_v11.tif \
        --data_output $OUTPUT_PATH \
        --model_path weights/deepglobe_buildings.pt \
        --cell_size 200 \
        --stride_size 190 \
        --tile_size 650 
        


Dependencies
-----------
- rasterio
- rio-tiler
- shapely
- fiona
- geopandas
- cw-tiler
- rtree


License
-------

See `LICENSE.txt <LICENSE.txt>`__.

Authors
-------

See `AUTHORS.txt <AUTHORS.txt>`__.

Changes
-------

See `CHANGES.txt <CHANGES.txt>`__.
