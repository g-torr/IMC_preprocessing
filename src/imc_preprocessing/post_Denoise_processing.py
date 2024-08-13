import numpy as np
import skimage
import skimage.io
import seaborn as sns
import matplotlib.pyplot as plt
import glob
import pandas as pd
import shutil
import os
from skimage import (
    data, restoration, util
)
from pathlib import Path

import tifffile as tp
def process_Carboplatin(config):
    '''Images from core are set with carboplatin =0. Moreover, the 99 percentile of the core images is used to normalise the images'''
    '/Carboplatin.tiff'
    input_path = os.path.join( config['root_data_folder'],config['tiff_folder_name_split'])
    biosamples_path  = os.path.join(config['root_data_folder'],config['biosamples_file'])
    file_list = file_list_from_img_folder(input_path,biosamples_path=biosamples_path)
    file_list['path'] = file_list['path']+'/Carboplatin.tiff'
    #getting the 95% of core images. Also saving Carboplatin to 0
    data = []
    img_core = []
    # identify the samples that we know should not have carboplatin to learn the baseline level of that marker in negative samples. 
    #From exploration I can see that even in Core, tissue contains some Platinum and I can see the shape of the tissue.
    samples_with_carboplatin = (file_list['SAMPLE_TYPE_(CORE/RESECTION)']=='RESECTION')*(file_list['NACT_treatment _group'].str.contains('carbo').fillna(False))
    for file in file_list[~samples_with_carboplatin].path:
        try:
            img = skimage.io.imread(file)
            img_core+=[img]
            data+=[np.quantile(img,q = 0.95)]
            output_file = file.replace('split_channels_nohpf','Img_Denoised/processed/')
            tp.imwrite(output_file,np.zeros(img.shape).astype('float32'))
        except FileNotFoundError:
            print(file,'not found')
            continue
    low_thr = np.quantile(data,q = 0.95)#take the quantile of the distribution of quantile. It is robust to up to 5% of resection misteakely being core.
    #Now treating resection, using the low_thr as the zero of the Carboplatin.
    #Find the 95% quantile of each image, taking the maximum across images to set the same  cutoff for normalisation
    data_res = []
    img_res=[]
    for file in file_list[samples_with_carboplatin].path:
        #load the distribution of the image
        img = skimage.io.imread(file)
        img_res+=[img]
        q = np.quantile(img,q = 0.95)
        data_res +=[q]
        
    max_data_res = max(data_res)
    for i, (img,file) in enumerate(zip(img_res,file_list[file_list['SAMPLE_TYPE_(CORE/RESECTION)']=='RESECTION'].path)):
        img_float = skimage.exposure.rescale_intensity(img,in_range=(low_thr,max_data_res))
        q = data_res[i]
        if q>1.2:
            #if the image has a lot of signal, make it sharper
            img = skimage.exposure.rescale_intensity(skimage.exposure.equalize_hist(img_float))
        else:
            # in this case, exposure.equalize_hist tends to overcorrect dark region overamplifying noise. Use a more noise 
            img = skimage.exposure.rescale_intensity(skimage.exposure.equalize_adapthist(img_float))
        output_file = file.replace('split_channels_nohpf','Img_Denoised/processed/')
        tp.imwrite(output_file,img)
    #copy Carboplatin also to 'Img_Denoised/non_prepocessed and 'Img_Denoised/rescaled
    for file in file_list.path:
        copy_from = file.replace('split_channels_nohpf','Img_Denoised/processed/')
        copy_to = file.replace('split_channels_nohpf','Img_Denoised/non_preprocessed/')
        shutil.copy(copy_from,copy_to)
        copy_to = file.replace('split_channels_nohpf','Img_Denoised/rescaled/')
        shutil.copy(copy_from,copy_to)
def process_all_channels_but_Cb(file_list,base_dir,config):
    def extract_quantile_from_img(f):
        '''take quantile per fov. Use log1p trasform'''
        img = skimage.io.imread(f)
        img = np.log1p(img[img>0])
        q = np.quantile(img,0.95)
        return q
    def load_and_rescale_by_quantile(f,q):
        img = skimage.io.imread(f)
        img = skimage.exposure.adjust_log(img)
        img = skimage.exposure.rescale_intensity(img, in_range=(0,q))
        return img    
    def acq_Id_from_path(path):
        return path.split('/')[-2]
    file_pattern = '*.tiff'  # Change to '*.tif' if your files have the '.tif' extension
    # Create a pattern to search for subdirectories with names starting with 'Leap'
    sub_dir_pattern = os.path.join(base_dir, 'Leap*')
    tb = pd.DataFrame(glob.glob(os.path.join(sub_dir_pattern, file_pattern), recursive=True),columns = ['filepath'])
    acq_mark = pd.DataFrame(list(tb.filepath.str.split('/').str[-2:]),columns = ['acquisition','marker'])
    acq_mark['marker'] = acq_mark.marker.str.replace('.tiff','')
    marker_list = list(acq_mark.marker.unique())
    marker_left_out = {'Carboplatin','DNA1','DNA2'}
    marker_list_stain = list(set(marker_list).difference({'Carboplatin'}))
    marker_list_row_norm = list(set(marker_list).difference(marker_left_out))

    #file_list = file_list[file_list.Keep == 'y']
    #process every channel independently, group images by staining date and normalise by quantile
    groupkey = 'Stain'
    all = {}
    
    for k, indx in file_list.groupby(groupkey).groups.items():
        thresholds = []
        for marker in marker_list_stain:
            paths = file_list.loc[indx].path+'/'+marker+'.tiff'
            paths = paths.to_list()
            quantile = np.median(skimage.io.ImageCollection(paths,load_func=extract_quantile_from_img))
            #normalise each image according to the quantile
            
            imgs = skimage.io.ImageCollection(paths,load_func=lambda f:load_and_rescale_by_quantile(f,quantile))
            for path_inp,img in zip(imgs.files,imgs):
                path_out = path_inp.replace('non_preprocessed','rescaled')#is the name of the file where to save
                Path(os.path.dirname(path_out)).mkdir(parents=True, exist_ok=True)#creates the folder if missing
                i = path_out.rfind('.')
                #path_out =  path_out[:i]+'_temp'+path_out[i:]#add _temp to the name of the file
                if not Path(path_out).is_file():
                    #if file does not exist, save it, otherwise skip
                    skimage.io.imsave(path_out,img)
            thresholds.append(quantile)
        all[k] = thresholds 

    #now convert_temp files so that the sum of markers in a Leap is always the same
    preprocessed_file_list = file_list_from_img_folder('/home/giuseppe/devices/Delta_Tissue/IMC/Img_Denoised/rescaled/',os.path.join(config['root_data_folder'],config['biosamples_file']))
    all_row = {}
    for k, indx in preprocessed_file_list.groupby('Leap_ID').groups.items():
        thresholds = []
        #generate thresholds
        for marker in marker_list_row_norm:
            paths = preprocessed_file_list.loc[indx].path+'/'+marker+'.tiff'
            paths = paths.to_list()
            #compute the median per ROI and then the median of all ROIs in a Leap
            thr = np.median(skimage.io.ImageCollection(paths,load_func=lambda x:np.median(skimage.io.imread(x))))
            thresholds.append(thr)
        thr = np.sum(thresholds)
        all_row[k] = thresholds
    
        for marker in marker_list_row_norm:
            paths = preprocessed_file_list.loc[indx].path+'/'+marker+'.tiff'
            paths = paths.to_list()
            imgs = skimage.io.ImageCollection(paths)
            for path_inp,img in zip(imgs.files,imgs):
                path_out = path_inp.replace('rescaled','processed')#is the name of the file where to save. Drop the _temp in the name
                Path(os.path.dirname(path_out)).mkdir(parents=True, exist_ok=True)#creates the folder if missing
                if True:#not Path(path_out).is_file():
                    #if file does not exist, save it, otherwise skip
                    img_processed = skimage.exposure.rescale_intensity(img/thr,in_range=(0,1/thr))
                    skimage.io.imsave(path_out,img_processed)
        for marker in set(marker_list_stain).difference(marker_list_row_norm):
            #process the DNAs
            #just copy the files
            paths = preprocessed_file_list.loc[indx].path+'/'+marker+'.tiff'
            paths = paths.to_list()
            for path_inp in paths:
                path_out = path_inp.replace('rescaled','processed')#is the name of the file where to save. Drop the _temp in the name
                shutil.copy(path_inp,path_out)


def file_list_from_img_folder(base_dir,biosamples_path):
    '''Create a table of path to acquisition folder, and the metadata related to that sample'''
    file_list = pd.Series(glob.glob(os.path.join(base_dir,'Leap*')))
    file_list = pd.DataFrame(file_list,columns = ['path'])
    file_list['acquisition_ID'] = file_list.path.str.split('/').str[-1]
    file_list['Leap_ID'] = file_list.acquisition_ID.str.split('_',n = 1).str[0].str.upper()
    file_list['Leap_ID'] = file_list['Leap_ID'].str[:7]#leap_ID should be Leap123, anything more is stripped
    #biosamples_path = '/home/giuseppe/devices/Delta_Tissue/IMC/IMC_data/ExtraDocs/processed_response_RCB.csv'
    biosamples =pd.read_csv(biosamples_path)
    file_list = file_list.merge(biosamples,left_on='Leap_ID',right_on= 'LEAP_ID').drop(['LEAP_ID'],axis = 1)#add metadata on patient
    return file_list
def loc_contrast_enhancement(base_dir):
    def load_and_enhance(name):
        img = skimage.io.imread(name)
        p2, p98 = np.percentile(img, (2, 98))
        img = skimage.exposure.rescale_intensity(img,in_range=(p2,p98), out_range = (0,.999))#added out_range to something <1 because otherwise rounding >1
        img = skimage.exposure.equalize_adapthist(img)
        return img
    new_directory = 'contrast_adj'
    file_pattern = '*.tiff'  # Change to '*.tif' if your files have the '.tif' extension
    # Create a pattern to search for subdirectories with names starting with 'Leap'
    sub_dir_pattern = os.path.join(base_dir, 'Leap*')
    paths = glob.glob(os.path.join(sub_dir_pattern, file_pattern), recursive=True)
    imgs = skimage.io.ImageCollection(paths,load_func=load_and_enhance)
    for origin_path,img_processed in zip(imgs.files,imgs):
        path_out = origin_path.replace('non_preprocessed',new_directory)
        Path(os.path.dirname(path_out)).mkdir(parents=True, exist_ok=True)#creates the folder if missing
        skimage.io.imsave(path_out,img_processed)
    #copy carboplatin
    file_list = file_list_from_img_folder(base_dir)
    file_list['path'] = file_list['path']+'/Carboplatin.tiff'
    for copy_from in file_list.path:
        copy_to = copy_from.replace('non_preprocessed',new_directory)
        shutil.copy(copy_from,copy_to)

def main():
    base_dir = '/home/giuseppe/devices/Delta_Tissue/IMC/Img_Denoised/non_preprocessed/'
    file_list = file_list_from_img_folder(base_dir)
    process_all_channels_but_Cb(file_list,base_dir)
    process_Carboplatin()
    loc_contrast_enhancement(base_dir)
if __name__=='__main__':
    main()