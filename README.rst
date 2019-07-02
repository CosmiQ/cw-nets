# This repository is no longer being updated. Future development of code tools for geospatial machine learning analysis will be done at https://github.com/cosmiq/solaris.

=========
cw-nets
=========


Segmentation Nets designed for use with SpaceNet datasets and other remote sensing data

An example of the output of this tool can be found at https://cwnets-demo.netlify.com/

Installation
------------
Using conda

Create Virtual Environment
```
conda create -n cw-nets python-3.6 pip cython
```

Install geospatial requirements
```
conda install --name cw-nets \
                    rtree \
                    gdal
```

Install Deep Learning Frameworks:
```
conda install pytorch torchvision cuda91 -c pytorch
conda install opencv scikit-image
```

Install CosmiQ tools
```
pip install git+https://github.com/CosmiQ/cw-tiler.git@dataset_creation
pip install git+https://github.com/CosmiQ/cw-nets.git@pytorch_generator
```






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
- cw-tiler https://github.com/CosmiQ/cw-tiler
- numpy
- tqdm
- shapely
- rasterio
- opencv
- scikit-image
- scikit-learn
- tensorflow
- keras
- torch
- torchvision



License
-------

See `LICENSE.txt <LICENSE.txt>`__.

Authors
-------

See `AUTHORS.txt <AUTHORS.txt>`__.

Changes
-------

See `CHANGES.txt <CHANGES.txt>`__.
