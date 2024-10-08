import numpy as np
import os
from os import PathLike
from pathlib import Path
from collections import defaultdict
import pandas as pd
from typing import Any, Dict, Generator, List, Optional, Sequence, Tuple, Union
from tqdm import tqdm
import shutil
import re
import argparse
from scipy.ndimage import maximum_filter
import tifffile
import logging
from imc_preprocessing import imcsegpipe
# this is a small variation of the repo
import logging
logger = logging.getLogger(__name__)

#from imcsegpipe.utils import sort_channels_by_mass
#import  imcsegpipe._imcsegpipe as imcsegpipe


def mcd_2_ome_tiff(config):
    '''This function extracts anything there is to extract from the mcd files and generate a metadata csv file.'''
    reprocess_mcd = True# False skip mcd extraction if tiff file already exists. Usefull if you have just ve added few leaps to collection
    root_data_folder = config['root_data_folder']

    path0 = os.path.join(root_data_folder,config['mcd_data_folder'])
    mcd_files_list = [str(el) for el in sorted(Path(path0).rglob("[!.]*.mcd"))]
    mcd_files_list = pd.Series(mcd_files_list)
    mcd_files_list = mcd_files_list[~mcd_files_list.str.contains(r"Large Pano MCD files|PanoramasCRUCKCI")]# these are all empty without acquisitions


    data_folder = ['/'.join(mcd_file.split('/')[:-1])for mcd_file in mcd_files_list]#take the path up to a level of the mcd file, it should contain the txt files

    tiff_files = [str(el) for el in sorted(Path(os.path.join(root_data_folder,config['tiff_folder_name_split'])).rglob("[!.]*.tiff"))]
    if reprocess_mcd:
        Leap_existing_files = []

    else:
        Leap_existing_files =        pd.Series(tiff_files).str.replace(os.path.join(root_data_folder,config['tiff_folder_name_split']),'').str.split('/').str[0].str.split('_').str[0].unique()

    acquisition_metadatas = []
    file_saved = []
    for mcd_file,main_folder in tqdm(list(zip(mcd_files_list,data_folder))):
        #main_folder is where the mcd files sit
        Leap_folder = mcd_file.replace(path0,'.').split('/',maxsplit = 2)[1] # take the highest directory inside path0 where the where the mcd file is
        #mcd_folder = '/'.join(mcd_file.split('/')[:-1])
        if mcd_file.replace(path0,'').split('/')[0] in Leap_existing_files:
            print('skipping'+mcd_file)
            continue
        #load metadata        
        try:
            acquisition_metadata = imcsegpipe.extract_mcd_file(mcd_file,acquisition_dir= os.path.join(path0,Leap_folder))#acquisition_dir is the path to the folder: Leap_ID
            #extract_mcd_file
            acquisition_metadatas.append(acquisition_metadata)
            
            file_saved+=(list(Leap_folder +'_'+acquisition_metadata.id.astype(str).values))
        except OSError:
            continue
    if reprocess_mcd:
        acquisition_metadata = pd.concat(acquisition_metadatas, copy=False)
        acquisition_metadata.to_csv(path0 +"/acquisition_metadata.csv",mode = 'w')

def ome_tiff_2_tiff(root_data_folder,tiff_folder_name_split,tiff_folder_name_combined,biosample_path):
    '''Extracts and saves the tiff files in appropriate folders. It also correct the names whenever appropriate'''
    tqdm.pandas()
    path0, code_2_Leap, path_tb,  metadata, panel = find_and_name_ome_tiff(root_data_folder,biosample_path)

    for _,file_row in tqdm(path_tb.iterrows()):
        #old_name = row['AcSession']+'_'+str(row['id'])
        # Pattern to match the specific format
        pattern = r"(.*)_s(\d)_a(\d{1,2})"
        
        # Find all occurrences of the pattern
        Leap_ID,_,acquisition_id = np.squeeze(re.findall(pattern, file_row.filename))
        row = metadata[(metadata.root==file_row.root)&(metadata.id==int(acquisition_id))&(metadata.AcSession == Leap_ID)]
        if len(row)>1:
            logger.warning('Multiple matches')
        row = row.iloc[0]
        description = row.description.lstrip('ROI_')
        if any( x in description.upper() for x in ['LASER','TEST']):
            # this is a test file, ignore acquisition 
            continue

        if '_' in description:
            code = description.split('_')[1]
            if any([ el in description for el in ['Leap067', 'Leap068']]):
                code = description.split('_')[0]  
            
            if '009_010_011' in Leap_ID :
                leap_9_10_11_mapper = {'19005858':'LEAP009','19005859':'LEAP011','19005860':'LEAP010'}#the sample id in the description is wrong
                Leap_ID = leap_9_10_11_mapper[code].capitalize()
                if str(row['id']) == '7':
                    # this is a repeated acquisition, skip it
                    continue
            else:
                if re.match("^\d{8}$",code):
                    #it is a 8 digits, looks like the code we want to use from biobank
                    try:
                        Leap_ID = code_2_Leap[code].capitalize()
                    except KeyError:
                        logger.warning('Potential problem in Leap'+Leap_ID+'. Skipping generating tiff file for row with description '+description )
                        pass
        if Leap_ID == 'Leap015_016':
            if 'TOP' in description:
                Leap_ID = 'Leap015'
            else:
                Leap_ID = 'Leap016'
        if Leap_ID == 'Leap017_018':
            if 'TOP' in description:
                Leap_ID = 'Leap017'
            else:
                Leap_ID = 'Leap018'
        if Leap_ID == 'Slide 42_MK_ROI':
            Leap_ID,row['id'] = description.split('_')
        if Leap_ID =='Leap091':
            #Claudia said that is to be removed
            continue
        if (Leap_ID =='Leap091') and (str(row['id']) in ['8','9','10']):
            Leap_ID ='Leap092'#Claudia mention on Whatsapp this is to be renamed
        
        Leap_ID =  Leap_ID.replace(' ','').capitalize()#remove any space in the name and make in capitalised format
        new_name = Leap_ID+'_'+str(row['id'])

        img = tifffile.imread(file_row.path)
        if np.any(np.array(img.shape[1:])<128):
            #if image is not at least of 128 pixel per side, ignore it
            continue
        img= img[~panel.marker.isna()]
        output_path = os.path.join(root_data_folder,tiff_folder_name_combined,Leap_ID)#
        logger.info('Saving in '+output_path)
        if not os.path.exists(output_path):
            #if folder does not exist, create it
            os.makedirs(output_path)

        tifffile.imwrite( 
        os.path.join(output_path,new_name+'.tiff'),
        data=img[np.newaxis, np.newaxis, :, :, :, np.newaxis],
        imagej=img.dtype in (np.uint8, np.uint16, np.float32),
        )
        #save in  the split_channels
        output_path_split = os.path.join(root_data_folder,tiff_folder_name_split,new_name)# eg. ../split_channels/Leap001_10'        
        logger.info('Saving in '+output_path)       
        if not os.path.exists(output_path_split):
            #if folder does not exist, create it
            os.makedirs(output_path_split)
        for channel,marker in list(zip(img,panel.dropna(axis = 0).marker.values)):
            tifffile.imwrite( 
            os.path.join(output_path_split,marker+'.tiff'),
            data=channel
            )
        panel.to_csv(path0+'/panel.csv')

def find_and_name_ome_tiff(root_data_folder,biosample_path):
    path0 = os.path.join(root_data_folder,'IMC_data')
    biobank = pd.read_csv(biosample_path)
    #biobank.dropna(axis = 0,inplace=True)
    biobank['code'] = biobank['BIOBANK_ID_num'].astype(str)
    code_2_Leap = biobank[['LEAP_ID','code']].drop_duplicates().set_index('code')['LEAP_ID'].to_dict()

    ome_tiff_paths = list(Path(path0).rglob("acquisition/[!.]*.ome.tiff"))
    path_tb = pd.DataFrame(ome_tiff_paths,columns = ['path'])
    path_tb['filename'] = path_tb['path'].apply(lambda x:x.name).str.rstrip('_ac.ome.tiff')
    path_tb['root'] = get_root(path0, path_tb)#return the path where the acquisition folder is, this is to match metadata['root'] below
    metadata = pd.read_csv(path0+'/acquisition_metadata.csv')
    #make root the path to the folder containing the mcd file
    metadata ['root'] = metadata.source_path.str.lstrip(  os.path.join(os.getcwd(),root_data_folder,'IMC_data')).str.split('/').str[0]
    metadata['description'] = metadata['description'].str.replace('170469269','17046926')#correct code for Leap54
    panel_file = path_tb.iloc[0].path
    panel_file = panel_file.with_name(panel_file.name[:-9]+'.csv')
    panel = pd.read_csv(panel_file)
    panel['marker'] = panel[panel.channel_label.str.contains('-')].channel_label.str.split('-',n = 1).str[1]
    panel.loc[panel['channel_name'] == 'Pt195','marker']='Carboplatin'
    return path0,code_2_Leap,path_tb,metadata,panel

def get_root(path0, path_tb):
    '''Remove the trailing path and return  the folder where "acquisition" folder is'''
    return path_tb.path.astype(str).str.lstrip(path0).str.split('/acquisition/').str[0]

#rename files and correct according to Leor table
def rename_leap_id(root_data_folder,tiff_folder_name_split):
    def leap3_4():
        swap_from_leap4_to_leap3 =[14,15,16]
        swap_from_leap3_to_leap4 =[11,12,13]
        #old_name:new_name
        file_2_rename = {'Leap003_'+str(id):'Leap004_'+str(id) for id in swap_from_leap3_to_leap4}|{'Leap004_'+str(id):'Leap003_'+str(id) for id in swap_from_leap4_to_leap3}
        files_2_delete = []    
        return files_2_delete,file_2_rename

    def leap17_18():
        files_2_delete = ['Leap018_'+str(x)for x in range(14,25)] 
        return files_2_delete,{}
    def leap9_11():
        files_2_delete = ['Leap009_'+x for x in ['1','2','3','4','5']]
        #swap_from_leap11_to_leap10 =['6','8','9','10','11','12','13']
        #swap_from_leap10_to_leap11 =['1','2','3','4','5']
        #file_2_rename = {'Leap011_'+str(id):'Leap010_'+str(id) for id in swap_from_leap11_to_leap10}|{'Leap010_'+str(id):'Leap011_'+str(id) for id in swap_from_leap10_to_leap11}
        return files_2_delete,{}
    def leap24_25():
        files_2_delete = ['Leap024_'+str(x) for x in range(1,7)]+['Leap025_7']
        return files_2_delete,{}    
    def leap40():
        files_2_delete = ['Leap040_'+str(x) for x in [2,3,4,5,7]]
        return files_2_delete,{}
    def leap91_92():
        file_2_rename = {'Leap092_'+str(id):'Leap091_'+str(id) for id in [8,9,10]}
        return [],file_2_rename
    def aggregate_changes():
        functions = [leap3_4,leap17_18,leap9_11,leap24_25,leap40,leap91_92]
        files_2_rename = {}
        files_2_delete = []
        for f in functions:
            a,b = f()
            files_2_delete+=a
            files_2_rename|=b
        return files_2_delete,files_2_rename

    files_2_delete,files_2_rename = aggregate_changes()
    output_path = os.path.join(root_data_folder,tiff_folder_name_split)
    for file in files_2_delete:
        try:
            shutil.rmtree(os.path.join(output_path,file))
        except FileNotFoundError:
            logging.warn(output_path+file+' not found')
    for old,new in files_2_rename.items():
        try:
            os.rename(os.path.join(output_path,old),os.path.join(output_path,new))
        except:
            logging.warn(os.path.join(output_path,old)+' not found')
def main(config):
    correct_extraction_options = ['no','all','mcd_2_ome_tiff','ome_tiff_2_tiff','rename_leap_id']
    root_data_folder = config['root_data_folder']

    if config['extraction'] not in correct_extraction_options:
        raise ValueError('Field "extraction" of config file '+args.config+' contains unrecognised value: '+config['extraction']+'. Please use instead one of the following options: '+','.join(correct_extraction_options) )
    if config['extraction']!='no':
        tiff_folder_name_split = config['tiff_folder_name_split']
        tiff_folder_name_combined = config['tiff_folder_name_combined']
        if (config['extraction']=='all') or (config['extraction']=='mcd_2_ome_tiff'):
            logger.info('Extracting ome-tiff from mcd ...')
            mcd_2_ome_tiff(config)
        if(config['extraction']=='all') or (config['extraction']=='ome_tiff_2_tiff'):
            logger.info('Extracting tiff from ome-tiff ...')
            ome_tiff_2_tiff(root_data_folder,tiff_folder_name_split,tiff_folder_name_combined,biosample_path = config['biosamples_file'])

        if(config['extraction']=='all') or (config['extraction']=='rename_leap_id'):
            logger.info('Renaming leap id ...')
            rename_leap_id(root_data_folder,tiff_folder_name_split)# it does not correct for the files in the combined tiff folder
    else:
        logger.info('Skipping extraction')
if __name__=='__main__':
    '''While it is recommended to run from main.py, this can be run as a standalone script using the config.yaml configuration '''
    from  imc_preprocessing.utils import load_config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml', help='Configuration file')
    args = parser.parse_args()
    config = load_config(args.config)

    main(config)

