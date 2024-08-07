import os
#!/usr/bin/env python
# coding: utf-8

# # IMC-Denoise: a content aware denoising pipeline to enhance imaging mass cytometry

# Here we will show an example for denoising the images with marker CD14 from our own human bone marrow IMC dataset. 

import numpy as np
import matplotlib.pyplot as plt
from IMC_Denoise.DeepSNiF_utils.DeepSNiF_DataGenerator import DeepSNiF_DataGenerator, load_training_patches
from IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF
import os
import argparse
import logging
import logging.handlers
# Configure file handler with rotation
file_handler = logging.handlers.RotatingFileHandler('log_denoise.log', maxBytes=1048576, backupCount=5, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
# Create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Add handlers to logger
logger.addHandler(file_handler)

import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))





def generate_patches(channel_name,Raw_directory,Save_directory,n_neighbours,n_iter,window_size ):
    '''Generate images patches and save them to disk'''

    # ### Training data preparation
    # Next, we use our raw images to build a training set.
    # Note: 
    # 1. The channel name must be consistant with the corresponding channel name in the image file names. For example, in our dataset, CD14 is conjucted with 144Nd. If the images with marker CD14 need to be denoised, the channel name will be set as its isotope name "144Nd".
    # 2. Raw_directory is the folder of all the raw images used for generating training set. Its subfolders are the imagesets of different tissues. The subfolders contains the images from all the channels of the same tissue. 
    # <b><br>Data_structure example:
    # <b><br>|---Raw_image_directory
    # <br>|---|---Tissue1
    # <br>|---|---|---Channel1_img.tiff
    # <br>|---|---|---Channel2_img.tiff
    # <br>             ...
    # <br>|---|---|---Channel_n_img.tiff
    # <br>|---|---Tissue2
    # <br>|---|---|---Channel1_img.tiff
    # <br>|---|---|---Channel2_img.tiff
    # <br>             ...
    # <br>|---|---|---Channel_n_img.tiff
    # <br>             ...
    # <br>|---|---Tissue_m
    # <br>|---|---|---Channel1_img.tiff
    # <br>|---|---|---Channel2_img.tiff
    # <br>             ...
    # <br>|---|---|---Channel_n_img.tiff
    # </b>
    # 3. Save_directory is the folder used for saving generated training data. If None, it will be saved in the default folder. For CD14, the saved training set is "training_set_144Nd.npz".
    # 4. n_neighbour and n_lambda are the parameters from DIMR algorithm for hot pixel removal in the training set generation process. 4 and 5 are their defaults. If the defaults are changed, the corresponding parameter should be declared in DeepSNiF_DataGenerator(). Otherwise, they can be omitted.
    # 5. The DeepSNiF_DataGenerator class search all the CD14 images in raw image directory, split them into multiple 64x64 patches, and then augment the generated data. Note the very sparse patches are removed in this process.
    # 6. Here we will save the generated training set and later reload it.

    # Release memory
    if 'generated_patches' in globals():
        del generated_patches
    
     
    DataGenerator = DeepSNiF_DataGenerator(channel_name = channel_name, n_neighbours = n_neighbours, n_iter = n_iter,window_size = window_size )
    generated_patches = DataGenerator.generate_patches_from_directory(load_directory = Raw_directory)
    if DataGenerator.save_patches(generated_patches, save_directory = Save_directory):
        print('Data generated successfully!')










def load_and_train(channel_name,deepsnif, Save_directory):
    '''
    Load the generated training data from directory
    '''
    saved_training_set = 'training_set_'+channel_name+'.npz'
    train_data = load_training_patches(filename = saved_training_set, save_directory = Save_directory)
    print('The shape of the loaded training set is ' + str(train_data.shape))
    train_loss, val_loss = deepsnif.train(train_data)
    return train_loss, val_loss

def main_train(config):
    root_data_folder = config['root_data_folder']
    tiff_folder_name_split = config['tiff_folder_name_split']
    Raw_directory = os.path.join(root_data_folder,tiff_folder_name_split)#directory for  raw images input that IMC_Denoise is trained on
    weights_path = os.path.join(config['IMC_Denoise']['Save_directory'],config['IMC_Denoise']['weights_folder'])
    channels_to_exclude = config['IMC_Denoise']['channels_to_exclude'].split(',')# convert csv field into list
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    # Load one acquisition directory
    acquisition_dirs = [d for d in os.listdir(Raw_directory) if os.path.isdir(os.path.join(Raw_directory, d))]
    if acquisition_dirs:
        # Take the first directory in the list
        first_acq_dir = os.path.join(Raw_directory, acquisition_dirs[0])

        # Get a list of channels by removing the file extension from files in the first directory
        channel_list = [os.path.splitext(file)[0] for file in os.listdir(first_acq_dir) if file.endswith('.tiff')]
        channel_list = [channel for channel in channel_list if channel not in channels_to_exclude]
       #remove unwanted channels 
    else:
        raise ValueError('No file found in '+Raw_directory)
    if config['IMC_Denoise']['retrain']:
        #model is trained for all markers, if it already exists, it is retrained
        markers_already_processed = []
    else:
        #if a folder already exists with that marker, do not retrain  
        markers_already_processed = [file.replace('weights_', '').replace('.keras', '') for file in os.listdir(weights_path) if file.startswith('weights_') and file.endswith('.keras')]
    #Parameters of IMC_Denoise
    params = config['IMC_Denoise']['params']
    Save_directory = config['IMC_Denoise']['Save_directory']
    
    # Iterate over marker_list and execute foo.py for each channel_name
    for channel_name in channel_list:
        if channel_name not in markers_already_processed:
            # Assuming foo.py is in the same directory as this script
            weights_name = "weights_"+channel_name+".keras"

            generate_patches(channel_name,Raw_directory=Raw_directory,
                            Save_directory=Save_directory,n_neighbours=params['n_neighbours'],
                            n_iter = params['n_iter'] ,window_size = params['window_size'] )
            deepsnif = DeepSNiF(train_epoches = params['train_epoches'], 
                            train_learning_rate = float(params['train_learning_rate']),
                            train_batch_size = params['train_batch_size'],
                            mask_perc_pix = float(params['pixel_mask_percent']),
                            val_perc = float(params['val_set_percent']),
                            loss_func = params['loss_function'],
                            weights_name = weights_name,
                            loss_name = params['loss_name'],
                            weights_dir = weights_path, 
                            is_load_weights = params['is_load_weights'],
                            lambda_HF = float(params['lambda_HF']))

            train_loss, val_loss = load_and_train(channel_name,deepsnif = deepsnif, Save_directory =  Save_directory)
            np.save(os.path.join(Save_directory,'trained_weights','loss_tv_'+channel_name+'npy'),[train_loss, val_loss])
            del deepsnif
            tf.keras.backend.clear_session() #to free up memory


