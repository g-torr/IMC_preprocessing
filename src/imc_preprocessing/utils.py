import yaml
import pathlib
import numpy as np
import pandas as pd
import datetime

def load_config(filename):
	"""
	Loads configuration from a YAML file.

	Args:
		filename (str): Path to the configuration YAML file.

	Returns:
		dict: Dictionary containing the loaded configuration.
	"""
	with open(filename, 'r') as f:
		return yaml.safe_load(f)
def format_sample_metadata(config_generic,config_metadata):
	'''
	Generate a file containing the metadata in a machine readable format. It aggregates the metadata about response and the staining information
	'''


	def convert_to_string(value):
		"""Converts datetime objects to 'day/month/year' string, and returns other values as is."""
		if pd.isna(value):
			return value
		elif isinstance(value, datetime.datetime):
			return value.strftime('%d/%m/%Y')
		else:
			return str(value)
		
	batches = pd.read_excel(config_metadata['batch'],index_col=0,converters={ 'DATE STAINED ': None})
	batches.drop('OUTCOME    (RESPONDER, NON-RESPONDER)',axis = 1,inplace=True)# use metadata table insteas
	batches = batches.rename_axis(index = 'LEAP_ID').reset_index()
	batches.columns = batches.columns.str.strip(' ').str.replace(' ','_')#remove trailing spaces in the column header and use _ instead of spaces
	batches['DATE_STAINED'] = batches['DATE_STAINED'].ffill()
	batches['DATE_STAINED'] = batches['DATE_STAINED'].apply(lambda x: convert_to_string(x))# column contains dates and string, pandas converts into a mixed data type, so I force it to be string
	batches['Stain_group'] = pd.factorize(batches['DATE_STAINED'])[0]# creates a numeric category of staining date. 

	#Now process metadata file and merge with batches
	metadata = pd.read_excel(config_metadata['metadata'],dtype={'biobank_id': str})
	metadata = metadata.dropna(axis = 1,how = 'all')
	metadata = metadata.drop(['Registration status','h&e', 'st', 'imc', 'la_icp_ms', 'flim'],axis = 1)
	metadata = metadata.rename({'leap_id':'LEAP_ID','analysis_id':'Patient','sample_type':'SAMPLE_TYPE','response':'Response','General comments':'Comments'},axis = 1)
	metadata = metadata.rename(columns = {'biobank_id':'BIOBANK_ID_num'})
	metadata['LEAP_ID'] = metadata['LEAP_ID'].str.upper() # our pipeline  works with upper LEAP_ID
	metadata['SAMPLE_TYPE'] = metadata['SAMPLE_TYPE'].str.upper()
	metadata = metadata.merge(batches.drop(['GST_ID'],axis = 1),left_on='LEAP_ID',right_on='LEAP_ID')
	metadata['SLIDE_NUMBER'] = metadata['SLIDE_NUMBER'].fillna(metadata['slide_num'])# try to fill using info from both datasets
	metadata.drop('slide_num',axis =1,inplace = True)
	metadata['BIOBANK_ID_num'] = metadata.BIOBANK_ID_num.fillna(metadata.BIOBANK_ID.str.split('-').str[0])
	#the columns have swapped ids for Leap3 and 4, this is due to the fact that the metadata inside the mcd files are swapped. This choise compensate that
	metadata['for_prediction'] = ~metadata['NACT_treatment _group'].isna()#patients for which we do not have the treatment, are those that did not receive neither ECT or CET carbo, but a different treatment plan. 
	#this means that they should not be used for prediction
	metadata['Response'] = metadata['Response'].fillna(metadata['old_response'])# for those who are in a different treatment plan, use the old  definition of response
	metadata['new_Response'] = np.where(metadata['extreme_non_responder'].isna(),metadata['Response'],'enR')# new response column contains: 'Non-Responder', 'pCR', 'enR', 'Responder' 
	#save data
	output_folder =pathlib.Path(config_generic['biosamples_file']).parent.resolve()
	output_folder.mkdir(parents=True, exist_ok=True)
	metadata.to_csv(config_generic['biosamples_file'],index = False)
