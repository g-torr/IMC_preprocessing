# IMC preprocessing
Extraction and preprocessing pipeline for IMC data. From a single mcd file, it extracts the tiff files using [steinbock](https://bodenmillergroup.github.io/steinbock/)  and run  [IMC denoise](https://github.com/PENGLU-WashU/IMC_Denoise) for denoising. To reduce batch effects, we also recommend applying contrast adjustment through Contrast Limited Adaptive Histogram Equalization.  
## Installation
Install the dependency IMC Denoise
```
$ git clone https://github.com/g-torr/IMC_Denoise.git
$ cd IMC_Denoise
$ pip install -e .
```
- Download the repo and install the package in your folder.
```
$ git clone https://github.com/g-torr/IMC_preprocessing.git
$ cd IMC_preprocessing
$ pip install -e .
```
## How to use it
### Quickstart
First, edit the file `/scripts/configs/config.yaml`. This file set the parameters for the pipeline. Make sure that `root_data_folder` points to the folder that contains the mcd files, with the relative path pointing from  `scripts/` folder. 
The general usage is: 
```
$ python scripts/main.py
```

## Directory structure of raw IMC images
The configuration file assumes that mcd data ar located in the folder `mcd_data_folder = IMC_data`, containing a structure like:
```
|---IMC_data
|---|---Leap001
|---|---|---Leap001.mcd
             ...
|---|---Leap002
|---|---|---Leap002_x.mcd
|---|---|---Leap002_y.mcd
|---IMC_preprocessing
|---|---|scripts|main.py
```
Each mcd file may contain several acquisitions. Acquisitions are saved in `tiff_folder_name_split` and `tiff_folder_name_combined` as grayscale and multichannel tiffs respectively. Here `a` and `b` represents acquisition ids.

```
|---IMC_data
|---IMC_preprocessing
|---tiff_folder_name_split
|---|---Leap001_a
|---|---|---|channel1.tiff
             ...
|---|---|---|channeln.tiff
|---|---Leap002_x_b
|---|---|---|channel1.tiff
             ...
|---|---|---|channeln.tiff

|---tiff_folder_name_combined
|---|---Leap001
|---|---|---Leap001_a.tiff
```
## Directory structure of images for  IMC denoise
IMC Denoise takes the images from `tiff_folder_name_split` folder, trains and predicts the new processed images to the `output_directory`. Parameters of IMC Denoise can be set up in the file `/scripts/configs/config.yaml`
