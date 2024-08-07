from  imc_preprocessing.utils import load_config
from imc_preprocessing import Denoise_train, Denoise_predict
import argparse
import logging
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config_file = load_config(args.config)
    ##----From mcd to tiff-------
    if config_file['extraction']!='no':
        import imc_preprocessing.reformatting_all_files as extraction
        extraction.main(config_file)
    else:
        logging.info('Skipping extraction')
    #-----IMC_Denoise---
    Denoise_train.main_train(config_file)
    Denoise_predict.main(config_file)
# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
    main()
