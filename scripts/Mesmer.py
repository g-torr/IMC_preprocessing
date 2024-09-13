
import pandas as pd
import numpy as np
import os
import scipy.ndimage as ndimage
from  imc_preprocessing.utils import load_config
import glob
import tqdm
from skimage.io import imread
import argparse
from matplotlib import pyplot as plt
from pathlib import Path
import tifffile

import logging
logging.basicConfig( level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def almost_fuzzy_logic(img_stack,k):
    '''take the k brightest channel per pixel,and average them.
    Note: if k == 0, it corresponds to the np.mean(axis = 0) functiom
    '''
    return np.sort(img_stack,axis = 0)[-k:,:,:].mean(axis = 0)
def load_imgs_and_concatenate(tb_subset,n_top_channels = None,smoothing_factor=0):
    '''Load a series of images, rescale, and adds them up '''
    if n_top_channels is None:
        function = lambda x:almost_fuzzy_logic(x,0)
        #gives the np.mean(,axis = 0)
    else:
        function = lambda x:almost_fuzzy_logic(x,n_top_channels)
    img_out_stack = np.zeros((len(tb_subset),*imread(tb_subset.filepath.iloc[0]).shape))
    for ind,filepath in enumerate(tb_subset.filepath):
        img = imread(filepath)

        img = ndimage.gaussian_filter(img,sigma = smoothing_factor)
        img_flattened = img.reshape(-1) 
        img_percentile = np.quantile(img_flattened,  q= .99)
        img =  img/img_percentile
        img[img>1]=1
        img_out_stack[ind] = img
        
    return function(img_out_stack)


def create_image_file_record(tiff_dir,membrane_markers,nucl_markers):


    file_pattern = '*.tiff'  # Change to '*.tif' if your files have the '.tif' extension

    # Create a pattern to search for subdirectories with names starting with 'Leap'
    sub_dir_pattern = os.path.join(tiff_dir, 'Leap*')
    tb = pd.DataFrame(Path(tiff_dir).expanduser().glob('Leap*/'+file_pattern),columns = ['filepath'])
    tb['acquisition_ID'] = tb['filepath'].map(lambda x:x.parts[-2])
    tb['marker'] = tb['filepath'].map(lambda x:x.stem)
    tb = tb[tb.marker.isin(membrane_markers+nucl_markers)].reset_index(drop = True)#restrict to segmentation markers only

    tb['membrane'] = tb.marker.isin(membrane_markers)
    return tb
def segment(tb, mask_path, deepcell_out_path, overwrite):
    from deepcell.applications import Mesmer
    app = Mesmer()
    for key,ind in tqdm.tqdm(tb.groupby('acquisition_ID').groups.items()):
        nucl_save_path = Path(deepcell_out_path)/(key+'_nuclear.tiff')
        memb_save_path = Path(deepcell_out_path)/(key+'_whole_cell.tiff')
        if ((os.path.exists(nucl_save_path))& os.path.exists(memb_save_path))&(~overwrite):
            #if cells already segmented, skip
            continue
        cond = tb.loc[ind].membrane #take membrane marker
        #img_memb = load_imgs_and_concatenate(tb.loc[ind][cond],smoothing_factor=1,function = almost_fuzzy_logic)
        #img_nucl = load_imgs_and_concatenate(tb.loc[ind][~cond],smoothing_factor=1)
        img_memb = load_imgs_and_concatenate(tb.loc[ind][cond],smoothing_factor=0)
        '''
        if np.any(np.array(img_memb.shape)<256):
            #if the image is not at least 256 x256, move on
            print(key, 'is small')
            continue
        '''
        img_nucl = load_imgs_and_concatenate(tb.loc[ind][~cond],smoothing_factor=0)
        image = np.stack((img_nucl,img_memb),axis =-1)#stack nuclear and membrane channel on the last axis
        X_train = np.expand_dims(image,axis = 0)
        

        try:
            segmentation_predictions = app.predict(X_train, image_mpp=1,compartment ="both")[0]#the [0] is because of the expand_dims  in the line before
        except :
            logger.warning('Segmentation for file '+key+' failed.')
            continue
        memb_save_path.parents[0].mkdir(parents=True, exist_ok=True)#create directory where saving segmentation mask if doesn't exist
        tifffile.imwrite(Path(mask_path)/(key+'.tiff'),image)#save the combined channels image        
        whole_cell_segmentation_prediction,nucl_segm_prediction = segmentation_predictions[:,:,0],segmentation_predictions[:,:,1]
        tifffile.imwrite(nucl_save_path,nucl_segm_prediction)
        tifffile.imwrite(memb_save_path,whole_cell_segmentation_prediction)




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)

    membrane_markers = ['CD14', 'CD11b', 'CD45', 'CD366', 'CD4', 'E-Cadherin',
        'HLA-DR-DQ-DP', 'CD8a', 'Beta-Catenin', 'B7-H4', 'CD3', 'CD27',
        'CD45RO', 'Vimentin']
    nucl_markers = ['DNA1','DNA2']
    tiff_dir =  Path(config['Processing']['output_directory']).expanduser()
    tb = create_image_file_record(tiff_dir=tiff_dir ,membrane_markers=membrane_markers,nucl_markers=nucl_markers)# pandas dataframe containing the path to tiff files for nuclear and membrane markers
    mask_path = Path(config['Mesmer']['mask_path']).expanduser()
    deepcell_out_path = Path(config['Mesmer']['deepcell_out_path']).expanduser()
    overwrite = config['Mesmer']['overwrite']

    if not os.path.exists(deepcell_out_path):
        os.makedirs(deepcell_out_path)

    if not os.path.exists(mask_path):
        os.makedirs(mask_path)
    if ~config['Mesmer']['gpu']:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'#use cpu if gpu is not able to handle

    logger.info('Starting segmentation')
    segment(tb, mask_path, deepcell_out_path, overwrite)# run Mesmer for cell segmentation
    logger.info('Finished segmentation.')

# set up file paths
# Entry point for the script, parses arguments and loads configuration
if __name__ == '__main__':
    main()
