from  imc_preprocessing.utils import load_config
from imc_preprocessing import Denoise_train
import argparse
import logging
import os
from IMC_Denoise.IMC_Denoise_main.DeepSNiF import DeepSNiF

logger = logging.getLogger(__name__)
def test_arguments(**args):
    print(args[['IMC_Denoise']])
    pass
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../scripts/configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    ##----From mcd to tiff-------
    import imc_preprocessing.reformatting_all_files as extraction
    print(config)
    ### IMC denoise
    root_data_folder = config['root_data_folder']
    tiff_folder_name_split = config['tiff_folder_name_split']
    Raw_directory = os.path.join(root_data_folder,tiff_folder_name_split)#directory for  raw images input that IMC_Denoise is trained on
    weights_path = os.path.join('..',config['IMC_Denoise']['weights_folder'])
    if not os.path.exists(weights_path):
        os.mkdir(weights_path)
    # Load one acquisition directory
    acquisition_dirs = [d for d in os.listdir(Raw_directory) if os.path.isdir(os.path.join(Raw_directory, d))]
    if acquisition_dirs:
        # Take the first directory in the list
        first_acq_dir = os.path.join(Raw_directory, acquisition_dirs[0])

        # Get a list of markers by removing the file extension from files in the first directory
        marker_list = [os.path.splitext(file)[0] for file in os.listdir(first_acq_dir) if file.endswith('.tiff')]
    else:
        raise ValueError('No file found in '+Raw_directory)
    if config['IMC_Denoise']['retrain']:
        #model is trained for all markers, if it already exists, it is retrained
        markers_already_processed = []
    else:
        #if a folder already exists with that marker, do not retrain  
        markers_already_processed = [file.replace('weights_', '').replace('.hdf5', '') for file in os.listdir(weights_path) if file.startswith('weights_') and file.endswith('.hdf5')]
    #Parameters of IMC_Denoise
    params = config['IMC_Denoise']['params']
    Save_directory = params['Save_directory']
    # Iterate over marker_list and execute foo.py for each channel_name
    for channel_name in marker_list:
        if channel_name not in markers_already_processed:
            # Assuming foo.py is in the same directory as this script
            weights_name = "weights_"+channel_name+".hdf5"

            test_arguments(channel_name=channel_name,Raw_directory=Raw_directory,
                            Save_directory=Save_directory,n_neighbours=params['n_neighbours'],
                            n_iter = params['n_iter'] ,window_size = params['window_size'] )

            deepsnif = DeepSNiF(train_epoches = params['train_epoches'], 
                            train_learning_rate = params['train_learning_rate'],
                            train_batch_size = params['train_batch_size'],
                            mask_perc_pix = params['pixel_mask_percent'],
                            val_perc = params['val_set_percent'],
                            loss_func = params['loss_function'],
                            weights_name = weights_name,
                            loss_name = params['loss_name'],
                            weights_dir = params['weights_save_directory'], 
                            is_load_weights = params['is_load_weights'],
                            lambda_HF = params['lambda_HF'])

    #-----IMC_Denoise---
    #Denoise.train()
# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
    main()
