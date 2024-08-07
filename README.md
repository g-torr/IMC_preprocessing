# IMC preprocessing
Extraction and preprocessing pipeline for IMC data. From a single mcd file, it extracts the tiff files using [steinbock](https://bodenmillergroup.github.io/steinbock/)  and run  [IMC denoise](https://github.com/PENGLU-WashU/IMC_Denoise) for denoising. 
## Installation
- Download the source code and install the package in your folder.
```
$ git clone https://github.com/g-torr/IMC_preprocessing.git
$ cd IMC_preprocessing
$ pip install -e .
```
## How to use it
### Quickstart
First, edit the file `/scripts/configs/config.yaml`. This file set the parameters for the pipeline. Make sure that `root_data_folder` points to the folder that contains the mcd files.
The general usage is: 
```
$ python scripts/main.py
```

## Directory structure of raw IMC images
```
|---Raw_image_directory
|---|---Sample1
|---|---|---Sample_name.mcd
             ...
|---|---Sample2
|---|---|---Sample_name_x.mcd
|---|---|---Sample_name_y.mcd
```
