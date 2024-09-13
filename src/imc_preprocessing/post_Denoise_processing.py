import numpy as np
import skimage
import skimage.io
import matplotlib.pyplot as plt
import glob
import pandas as pd
import shutil
import os
from pathlib import Path

import tifffile as tp
import logging
logger = logging.getLogger(__name__)
def process_Carboplatin(file_list,output_path):
    '''Images from core are set with carboplatin =0. Moreover, the 99 percentile of the core images is used to normalise the images'''
    file_list['path'] = file_list.path.map(lambda x:os.path.join(x,'Carboplatin.tiff'))
    #getting the 95% of core images. Also saving Carboplatin to 0

    data = []
    img_core = []
    # Exploring the images,  I can see that even in Core, tissue contains some Platinum and the distribution matches the shape of the tissue
    # I want to learn the baseline level of Carboplatin in tissues. For that,
    # I identify the samples that we know should not have carboplatin to learn the baseline level of that marker in negative samples. 

    samples_with_carboplatin = (file_list['SAMPLE_TYPE']=='RESECTION')*((file_list['NACT_treatment _group'].fillna('nan').str.contains('carbo')).astype(bool))
    for file in file_list[~samples_with_carboplatin].path:
        try:
            img = skimage.io.imread(file)
            img_core+=[img]
            data+=[np.quantile(img,q = 0.95)]
            relative_path = Path(file).relative_to(Path(file).parents[1])# take the acquisition folder and carboplatin.tiff file 
            output_file = Path(output_path)/relative_path # join the line above with the output directory, to create the new path
            output_file.parents[0].mkdir(parents=True, exist_ok=True)
            #output_file = file.replace('split_channels_nohpf','Img_Denoised/processed/')
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
    for i, (img,file) in enumerate(zip(img_res,file_list[samples_with_carboplatin].path)):
        img_float = skimage.exposure.rescale_intensity(img,in_range=(low_thr,max_data_res))# everything below low_thr is zero, everything above max_data_res is 1
        q = data_res[i]
        if q>1.2:
            #if the image has a lot of signal, make it sharper
            img = skimage.exposure.rescale_intensity(skimage.exposure.equalize_hist(img_float))
        else:
            # in this case, exposure.equalize_hist tends to overcorrect dark region overamplifying noise.
            img = skimage.exposure.rescale_intensity(skimage.exposure.equalize_adapthist(img_float))
        relative_path = Path(file).relative_to(Path(file).parents[1])# take the acquisition folder and carboplatin.tiff file 
        output_file = Path(output_path)/relative_path # join the line above with the output directory, to create the new path
        output_file.parents[0].mkdir(parents=True, exist_ok=True)# create all folders up to the parent of the file
        tp.imwrite(output_file,img)

def process_all_channels_but_Cb(file_list,base_dir,config):
    '''This function computes image normalisation and outputs in rescaled and processed folders.'''

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

    output_directory = config['Processing']['output_directory']
    output_directory_name = Path(output_directory).parts[-1]# take the name of the folder that contains Leapxyz_a, e.g. 'processed'
    scaled_output_directory = output_directory.replace(output_directory_name,'scaled')
    #file_list = file_list[file_list.Keep == 'y']
    #process every channel independently, group images by staining date and normalise by quantile in the staining group
    normalise_images_by_group(file_list, marker_list_stain, groupkey = 'Stain_group',output_path= scaled_output_directory ) 

    ## convert files from previous step so that the sum of markers in a Leap is always the same
    #process every channel independently, group images by leap and compute the median of the leap per channel.
    #Final images are such that the sum across markers of the median marker expression in a Leap is 1
    unit_vector_image_normalisation(marker_list_stain,groupkey = 'Leap_ID', marker_list_row_norm = marker_list_row_norm,config = config, input_dir = scaled_output_directory )

def normalise_images_by_group(file_list, marker_list_stain, groupkey,output_path):
    '''Normalise images by taking the 95% per stain group, indipendently for each channel'''
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

    for indx in file_list.groupby(groupkey).groups.values():
        
        for marker in marker_list_stain:
            paths = file_list.loc[indx].path+'/'+marker+'.tiff'
            paths = paths.to_list()
            quantile = np.median(skimage.io.ImageCollection(paths,load_func=extract_quantile_from_img))
            #normalise each image according to the quantile
            
            imgs = skimage.io.ImageCollection(paths,load_func=lambda f:load_and_rescale_by_quantile(f,quantile))
            for path_inp,img in zip(imgs.files,imgs):
                relative_path = Path(path_inp).relative_to(Path(path_inp).parents[1])# take the acquisition folder and tiff file 
                path_out = os.path.join(output_path,relative_path)  #join output_path with  Leap123_x/channel.tiff
                #it is the name of the folder where to save
                Path(os.path.dirname(path_out)).parents[0].mkdir(parents=True, exist_ok=True)#creates the folder if missing
                skimage.io.imsave(path_out,img)
        

def unit_vector_image_normalisation(marker_list_stain,groupkey, marker_list_row_norm, config,input_dir):

        #first load the images generated in the the previous step
    preprocessed_file_list = file_list_from_img_folder(input_dir,os.path.join(config['biosamples_file']))

    all_row = {}
    for k, indx in preprocessed_file_list.groupby(groupkey).groups.items():
        thresholds = []
        #generate thresholds
        for marker in marker_list_row_norm:
            paths = preprocessed_file_list.loc[indx].path+'/'+marker+'.tiff'
            paths = paths.to_list()
            #compute the median per ROI and then the median of all ROIs in a Leap
            thr = np.median(skimage.io.ImageCollection(paths,load_func=lambda x:np.median(skimage.io.imread(x))))
            thresholds.append(thr)
        thr = np.sum(thresholds)
        #take the sum across channels and use it to normalise each leap
        all_row[k] = thresholds
    
        for marker in marker_list_row_norm:
            paths = os.path.join(preprocessed_file_list.loc[indx].path,marker+'.tiff')
            paths = paths.to_list()
            imgs = skimage.io.ImageCollection(paths)
            for path_inp,img in zip(imgs.files,imgs):
                relative_path = Path(path_inp).relative_to(Path(path_inp).parents[1])# take the acquisition folder and tiff file 
                path_out = config['Processing']['output_directory']/relative_path # join the line above with the output directory, to create the new path
                Path(os.path.dirname(path_out)).parents[0].mkdir(parents=True, exist_ok=True)#creates the folder if missing
                img_processed = skimage.exposure.rescale_intensity(img/thr,in_range=(0,1/thr))
                skimage.io.imsave(path_out,img_processed)
        for marker in set(marker_list_stain).difference(marker_list_row_norm):
            #process the DNAs
            #just copy the files
            paths = preprocessed_file_list.loc[indx].path+'/'+marker+'.tiff'
            paths = paths.to_list()
            for path_inp in paths:
                relative_path = Path(path_inp).relative_to(Path(path_inp).parents[1])# take the acquisition folder and tiff file 
                path_out = config['Processing']['output_directory']/relative_path # join the line above with the output directory, to create the new path
                shutil.copy(path_inp,path_out)

def file_list_from_img_folder(img_folder,biosamples_path):
    '''
    Create a table of path to acquisition folder, and the metadata related to that sample.
    img_folder is the path to the folder containing all acquisitions, one in each folder.
    i.e.
    |img_folder|
    |--------|Leap001_8
    |--------|Leap001_8|DNA1.tiff'''
    file_list = pd.Series(glob.glob(os.path.join(img_folder,'Leap*')))
    file_list = pd.DataFrame(file_list,columns = ['path'])
    file_list['acquisition_ID'] = file_list.path.str.split('/').str[-1]
    file_list['Leap_ID'] = file_list.acquisition_ID.str.split('_',n = 1).str[0].str.upper()
    file_list['Leap_ID'] = file_list['Leap_ID'].str[:7]#leap_ID should be Leap123, anything more is stripped. Consider that there are cases like Leap088a
    biosamples =pd.read_csv(biosamples_path)
    file_list = file_list.merge(biosamples,left_on='Leap_ID',right_on= 'LEAP_ID').drop(['LEAP_ID'],axis = 1)#add metadata on patient
    return file_list
def loc_contrast_enhancement( input_dir,output_dir):
    '''It creates a folder in config['Processing']['output_directory'] and saves the finally processed images.
     Pixel intensity is rescaled from 2% and 98% and contrast is enhanced with CLAHE.'''
    def load_and_enhance(name):
        img = skimage.io.imread(name)
        p2, p98 = np.percentile(img, (2, 98))
        if np.any(np.isnan([p2,p98])):
            logger.warning('Nan in file '+name)
        img = skimage.exposure.rescale_intensity(img,in_range=(p2,p98), out_range = (0,.999))#added out_range to something <1 because otherwise rounding >1
        img = skimage.exposure.equalize_adapthist(img)
        return img

    file_pattern = '*.tiff'  # Change to '*.tif' if your files have the '.tif' extension
    # Create a pattern to search for subdirectories with names starting with 'Leap'
    sub_dir_pattern = os.path.join(input_dir, 'Leap*')# Leap folders that are in the denoised image folder
    paths = glob.glob(os.path.join(sub_dir_pattern, file_pattern), recursive=True)# iterator over all *.tiff images 
    paths = [path for path in paths if Path(path).stem!='Carboplatin']
    imgs = skimage.io.ImageCollection(paths,load_func=load_and_enhance)
    for path_inp,img_processed in zip(imgs.files,imgs):
        relative_path = Path(path_inp).relative_to(Path(path_inp).parents[1])# take the acquisition folder and tiff file 
        path_out = output_dir/relative_path # join the line above with the output directory, to create the new path
        Path(path_out).parents[0].mkdir(parents=True, exist_ok=True)#creates the folder if missing
        skimage.io.imsave(path_out,img_processed)
    '''
    #copy carboplatin
    file_list = file_list_from_img_folder(input_dir)
    file_list['path'] = file_list['path']+'/Carboplatin.tiff'
    for copy_from in file_list.path:
        copy_to = copy_from.replace(old_directory,new_directory)
        shutil.copy(copy_from,copy_to)
    '''
def main(config):
    input_dir = config['IMC_Denoise']['output_directory']# where all images but Carboplatin are loaded from
    output_dir = Path(config['Processing']['output_directory']).expanduser()#where images are saved to
    #This assumes Carboplatin is not processed by IMC_Denoise. 
    input_carbo = os.path.join( config['root_data_folder'],config['tiff_folder_name_split'])#input carboplatin channel

    biosamples_path  = os.path.join(config['biosamples_file'])
    file_list = file_list_from_img_folder(input_carbo,biosamples_path=biosamples_path)
    process_Carboplatin(file_list=file_list,output_path=  output_dir)

    file_list = file_list_from_img_folder(img_folder=input_dir,biosamples_path=biosamples_path)
    if str(config['Processing']['mode']).upper() =='CLAHE':
        logger.info('Proceeding with CLAHE')
        loc_contrast_enhancement(input_dir,output_dir)
    else:
        process_all_channels_but_Cb(file_list,input_dir,config=config)


if __name__=='__main__':
    main()