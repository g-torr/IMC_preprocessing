# -*- coding: utf-8 -*-

import numpy as np
import time
import gc
import os

import argparse
from os import listdir
from os.path import isfile, join, abspath, exists
from glob import glob
import tifffile as tp
from tensorflow.keras import optimizers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from IMC_Denoise.IMC_Denoise_main.DeepSNiF_model import DeepSNiF_net, DeepSNiF_net_small
from IMC_Denoise.IMC_Denoise_main.loss_functions import create_I_divergence, create_mse
from IMC_Denoise.IMC_Denoise_main.DIMR import DIMR
from IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF

from IMC_Denoise.Anscombe_transform.Anscombe_transform_functions import Anscombe_forward, Anscombe_inverse_exact_unbiased

def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

'''
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--channel_name", help = "the denoised channel name", type = str)
parser.add_argument("--load_directory", help = "the folder of the raw IMC images", type = str)
parser.add_argument("--save_directory", help = "the folder to save the denoised IMC images", type = str)
parser.add_argument("--loss_func", help = "the folder to save the denoised IMC images", type = str, default = "I_divergence")
parser.add_argument("--weights_name", help = "trained network weights. hdf5 format", type = str)
parser.add_argument("--weights_save_directory", help = "directory of trained network weights", type = str, default = None)
parser.add_argument("--batch_size", help = "batch size in prediction", type = int, default = 1)
parser.add_argument("--DIMR", help = "using DIMR?", default = True, type = str2bool)
parser.add_argument("--n_neighbours", help = "DIMR algorithm parameter", default = 4, type = int)
parser.add_argument("--n_iter", help = "DIMR algorithm parameter", default = 3, type = int)
parser.add_argument("--slide_window_size", help = "DIMR algorithm parameter", default=3, type = int)
parser.add_argument("--GPU", help = "using GPU?", default = True, type = str2bool)
parser.add_argument("--network_size", help = "normal or small network to be used?", default = 'small', type = str)
args = parser.parse_args()
print(args)
'''
def load_imgs(channel_name,Raw_directory,myDIMR):
    '''Load images and apply DIMR if selected'''


    Max_row_num = 0
    Max_col_num = 0
    Image_collect = []
    Img_folders = glob(join(Raw_directory, "*", ""))
    for Sub_img_folder in Img_folders:
        Img_list = [f for f in listdir(Sub_img_folder) if isfile(join(Sub_img_folder, f)) & (f.endswith(".tiff") or f.endswith(".tif"))]
        for Img_file in Img_list:
            if channel_name.lower() == os.path.splitext(Img_file)[0].lower():#strip extension to file and check it matches with marker
                Img_read = tp.imread(Sub_img_folder + Img_file).astype('float32')
                Img_read = myDIMR.perform_DIMR(Img_read)# I always run DIMR, as in the training 
                Image_collect.append(single_img_info(Img_read, Sub_img_folder, Img_file))
                Rows, Cols = np.shape(Img_read)
                Max_row_num = max(Max_row_num, Rows)
                Max_col_num = max(Max_col_num, Cols)
                break
                
    Max_row_num = int((Max_row_num//16+1)*16)
    Max_col_num = int((Max_col_num//16+1)*16)
    return Image_collect,Max_row_num,Max_col_num
def process_channel(channel_name,params,weights_path,Raw_directory,output_directory):
    '''This function runs the full prediction pipeline on all images of channel_name.
    It runs DIMR, it create a image collection where all images are in the same shape, 
    it runs deepsnif, and then inverse transforms images back to their original shape.
    '''
    weights_name = "weights_"+channel_name+".keras"
    myDIMR = DIMR(n_neighbours = params['n_neighbours'], n_iter = params['n_iter'], window_size = params['window_size'])
    deepsnif = DeepSNiF(train_epoches = params['train_epoches'], 
                train_learning_rate = float(params['train_learning_rate']),
                train_batch_size = params['train_batch_size'],
                mask_perc_pix = float(params['pixel_mask_percent']),
                val_perc = float(params['val_set_percent']),
                loss_func = params['loss_function'],
                weights_name = weights_name,
                loss_name = params['loss_name'],
                weights_dir = weights_path, 
                is_load_weights = True, # load trained model for prediction
                lambda_HF = float(params['lambda_HF']))
    Image_collect,Max_row_num,Max_col_num = load_imgs(channel_name,Raw_directory,myDIMR)

    All_img_read = transform_images(params, deepsnif, Image_collect, Max_row_num, Max_col_num)# images are padded to have the same shape
    All_img_denoised = deepsnif.trained_model.predict(All_img_read, batch_size = params['batch_size'])# apply Deepsnif correction
    finalise_denoised_images(params, Raw_directory, output_directory, deepsnif, Image_collect, All_img_denoised)
    #images are transformed back to their original shape and saved 


def transform_images(params, deepsnif, Image_collect, Max_row_num, Max_col_num):
    Img_num = len(Image_collect)
    All_img_read = np.zeros((Img_num, Max_row_num, Max_col_num, 1))
    for ii in range(Img_num):
        Cur_img = Image_collect[ii].Img
        Image_collect[ii].Img = None
        Rows, Cols = np.shape(Cur_img)
        
        if params['loss_function'] == 'mse_relu':
            Cur_img = Anscombe_forward(Cur_img)
            Cur_img = np.divide(Cur_img - 2*np.sqrt(3/8), deepsnif.range_val)
        else:
            Cur_img = np.divide(Cur_img, deepsnif.range_val)
        
        Rows_diff = Max_row_num - Rows
        Cols_diff = Max_col_num - Cols
        Rows_diff1, Rows_diff2 = split_border(Rows_diff)
        Cols_diff1, Cols_diff2 = split_border(Cols_diff)
            
        All_img_read[ii][:,:,0] = np.pad(Cur_img,((Rows_diff1,Rows_diff2),(Cols_diff1,Cols_diff2)),'edge')
        Image_collect[ii].Pad_dims = [Rows_diff1, Rows_diff2, Cols_diff1, Cols_diff2]
    return All_img_read

def finalise_denoised_images(params, Raw_directory, output_directory, deepsnif, Image_collect,  All_img_denoised):
    '''Transform images in the original channel space, cut images to fit in their original shape,
    and save them to disk'''
    Img_num = len(Image_collect)
    for ii in range(Img_num):
        Img_denoised = All_img_denoised[ii][:,:,0]
        Pad_dims = Image_collect[ii].Pad_dims
        Sub_img_folder = Image_collect[ii].Sub_folder 
        Img_name = Image_collect[ii].Img_name 

        Img_denoised = Img_denoised[Pad_dims[0]:(-Pad_dims[1]),Pad_dims[2]:(-Pad_dims[3])]

        if params['loss_function'] == 'mse_relu':
            Img_denoised = Img_denoised * deepsnif.range_val + 2*np.sqrt(3/8)
            Img_denoised = Anscombe_inverse_exact_unbiased(Img_denoised)
        else:
            Img_denoised = Img_denoised * deepsnif.range_val
        
        Img_denoised[Img_denoised<0] = 0
        sub_save_directory = join(output_directory, Sub_img_folder.lstrip(Raw_directory))#original code strip one caracter too many
        if not exists(sub_save_directory):
            os.makedirs(sub_save_directory)
        tp.imsave(join(sub_save_directory, Img_name), Img_denoised.astype('float32'))

        print(sub_save_directory + Img_name + ' saved!')

# define a class to save image information
class single_img_info:
    def __init__(self, Img = None, Sub_folder = None, Img_name = None, Pad_dims = None):
        self.Img = Img
        self.Sub_folder = Sub_folder
        self.Img_name = Img_name
        self.Pad_dims = Pad_dims

def split_border(length):
    half_length = int(length/2)
    if length%2 == 0:
        return half_length, half_length
    else:
        return half_length, half_length + 1

def main(config):
    params = config['IMC_Denoise']['params']
    if not str2bool(params['GPU']):
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if params['network_size'] == 'normal':
        network_used = DeepSNiF_net
    elif params['network_size'] == 'small':
        network_used = DeepSNiF_net_small
    else:
        raise Exception('The network_size should be either normal or small!')
    root_data_folder = config['root_data_folder']
    tiff_folder_name_split = config['tiff_folder_name_split']
    Raw_directory = os.path.join(root_data_folder,tiff_folder_name_split)
    #directory for  raw images input that IMC_Denoise is trained on
    channels_to_exclude = config['IMC_Denoise']['channels_to_exclude'].split(',')# convert csv field into list

    weights_path = os.path.join(config['IMC_Denoise']['Save_directory'],config['IMC_Denoise']['weights_folder'])
    output_directory = config['IMC_Denoise']['output_directory']
    # Load one acquisition directory
    acquisition_dirs = [d for d in os.listdir(Raw_directory) if os.path.isdir(os.path.join(Raw_directory, d))]

    if acquisition_dirs:
        # Take the first directory in the list
        first_acq_dir = os.path.join(Raw_directory, acquisition_dirs[0])
        
        # Get a list of markers by removing the file extension from files in the first directory
        channel_list = [os.path.splitext(file)[0] for file in os.listdir(first_acq_dir) if file.endswith('.tiff')]
        channel_list = [channel for channel in channel_list if channel not in channels_to_exclude]
    else:
        raise ValueError('No file found in '+Raw_directory)

    for channel_name in channel_list:
        start = time.time()
        process_channel(channel_name,params,weights_path,Raw_directory,output_directory)
        _ = gc.collect()
        end = time.time()
        print(end - start)
