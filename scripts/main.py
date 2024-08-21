from  imc_preprocessing.utils import load_config,format_sample_metadata
from imc_preprocessing import Denoise_train, Denoise_predict,post_Denoise_processing
import argparse
import logging
import os
import logging.handlers

logging.basicConfig(filename='logging.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config_file = load_config(args.config)
    if not os.path.isfile(config_file['biosamples_file']):
        #if the file containing the processed sample metadata is not found, create it
        config_metadata = load_config('configs/batch_metadata.yaml')
        format_sample_metadata(config_generic=config_file,config_metadata=config_metadata)
    ##----From mcd to tiff-------
    if config_file['extraction']!='no':
        import imc_preprocessing.extraction as extraction
        extraction.main(config_file)
    else:
        logging.info('Skipping extraction')
    #-----IMC_Denoise---
    if config_file['IMC_Denoise']['skip']:
        logging.info('Skippping IMC_Denoise')
        pass
    else:
        Denoise_train.main_train(config_file)
        Denoise_predict.main(config_file)
    post_Denoise_processing.main(config_file)
# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
    main()
