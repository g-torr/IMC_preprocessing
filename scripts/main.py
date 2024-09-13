from  imc_preprocessing.utils import load_config,format_sample_metadata
import argparse
import logging
import os
import logging.handlers
import gc

logging.basicConfig(filename='logging.log', level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')



def IMC_Denoise_transformation(config_file):
    if config_file['IMC_Denoise']['skip']:
        logging.info('Skippping IMC_Denoise')
        pass
    else:
        from imc_preprocessing import Denoise_train, Denoise_predict
        Denoise_train.main_train(config_file)
        gc.collect()
        Denoise_predict.main(config_file)
        gc.collect()
def extract_mcd_to_tiff(config_file):
    if config_file['extraction']!='no':
        import imc_preprocessing.extraction as extraction
        extraction.main(config_file)
        gc.collect()
    else:
        logging.info('Skipping extraction')

def process_response_metadata(config_file):
    if not os.path.isfile(config_file['biosamples_file']):
        #if the file containing the processed sample metadata is not found, create it
        config_metadata = load_config('configs/batch_metadata.yaml')
        format_sample_metadata(config_generic=config_file,config_metadata=config_metadata)

def post_denoise_transformation(config_file):
    if config_file['Processing']['skip']:
        logging.info('Skipping post denoise processing')
    elif config_file['Processing']['mode'] in ['CLAHE','unit']:
        from imc_preprocessing import post_Denoise_processing
        post_Denoise_processing.main(config_file)
    else:
        ValueError('Option '+config_file['Processing']['mode']+' not recognised')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config_file = load_config(args.config)
    process_response_metadata(config_file)
    ##----From mcd to tiff-------
    extract_mcd_to_tiff(config_file)
    #-----IMC_Denoise---
    IMC_Denoise_transformation(config_file)
    #----post denoise transformation----
    post_denoise_transformation(config_file)

# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
    main()
