import yaml
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
	import pandas as pd
	import datetime

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

	metadata = pd.read_excel(config_metadata['metadata'],dtype={'biobank_id': str})
	metadata = metadata.rename(columns = lambda x:x.upper()).dropna(axis = 1,how = 'all').drop(['REGISTRATION STATUS','H&E','ST','LA_ICP_MS','FLIM','IMC'],axis = 1)
	metadata['LEAP_ID'] = metadata['LEAP_ID'].str.upper() # our pipeline  works with upper LEAP_ID
	metadata['SAMPLE_TYPE'] = metadata['SAMPLE_TYPE'].str.upper()
	metadata = metadata.rename({'ANALYSIS_ID':'Patient'},axis = 1)
	metadata = metadata.rename(columns = {'BIOBANK_ID':'BIOBANK_ID_num'})
	metadata = metadata.merge(batches.drop(['GST_ID'],axis = 1),left_on='LEAP_ID',right_on='LEAP_ID')
	metadata['SLIDE_NUMBER'] = metadata['SLIDE_NUMBER'].fillna(metadata['SLIDE_NUM'])# try to fill using info from both datasets
	metadata.drop('SLIDE_NUM',axis =1,inplace = True)
	metadata.to_csv(config_generic['biosamples_file'])
