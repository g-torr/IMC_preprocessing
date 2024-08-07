from  imc_preprocessing.utils import load_config
from imc_preprocessing import Denoise_train
import argparse
import logging
import os


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
    print(Denoise_train)
    #-----IMC_Denoise---
    #Denoise.train()
# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
    main()
