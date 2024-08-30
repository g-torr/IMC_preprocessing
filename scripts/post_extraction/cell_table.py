import argparse
from ark.segmentation import marker_quantification
from alpineer import io_utils
from  imc_preprocessing.utils import load_config
import os
import logging
logger = logging.getLogger(__name__)

def cell_table(tiff_dir,cell_tb_path,deepcell_out_path):
    fovs = io_utils.list_folders(tiff_dir)


    # set to True to add nuclear cell properties to the expression matrix
    nuclear_counts = True

    # set to True to bypass expensive cell property calculations
    # only cell label, size, and centroid will be extracted if True
    fast_extraction = False
    # now extract the segmented imaging data to create normalized and transformed expression matrices
    # note that if you're loading your own dataset, please make sure all the imaging data is in the same folder
    # with each fov given its own folder and all fovs having the same channels
    cell_table_size_normalized, cell_table_arcsinh_transformed = \
    marker_quantification.generate_cell_table(segmentation_dir=deepcell_out_path,
                                              tiff_dir=tiff_dir,
                                              img_sub_folder=None,
                                              fovs=fovs,
                                              batch_size=20,
                                              nuclear_counts=nuclear_counts,
                                              fast_extraction=fast_extraction)





    if not os.path.exists(cell_tb_path):
        os.makedirs(cell_tb_path)
    
    # Set the compression level if desired, ZSTD compression can offer up to a 60-70% reduction in file size.
    # NOTE: Compressed `csv` files cannot be opened in Excel. They must be uncompressed beforehand.
    compression = None

    # Uncomment the line below to allow for compressed `csv` files.
    #compression = True#{"method": "zstd", "level": 3}

    if (compression is None) or (compression ==False):
        cell_table_normalized_path = os.path.join(cell_tb_path, 'cell_table_size_normalized.csv')
        cell_table_arcsinh_path = os.path.join(cell_tb_path, 'cell_table_arcsinh_transformed.csv')
    else:
        cell_table_normalized_path = os.path.join(cell_tb_path, 'cell_table_size_normalized.zstd')
        cell_table_arcsinh_path = os.path.join(cell_tb_path, 'cell_table_arcsinh_transformed.zstd')
    
    logger.info('Saving cell table at: '+cell_table_normalized_path)
    cell_table_size_normalized.to_csv(cell_table_normalized_path,
                                   index=False)
    cell_table_arcsinh_transformed.to_csv(cell_table_arcsinh_path,
                                       index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)
    tiff_dir =  config['Processing']['output_directory']
    cell_tb_path = config['Mesmer']['cell_tb_path']
    deepcell_out_path = config['Mesmer']['deepcell_out_path']
    cell_table(tiff_dir =tiff_dir,cell_tb_path=cell_tb_path,deepcell_out_path=deepcell_out_path)#compute cell table

