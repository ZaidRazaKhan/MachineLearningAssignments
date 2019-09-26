import warnings
import pandas as pd


def preprocess(data1):
	"""
	Preprocesses the hourly pollutant data 
	(adjust the rows and colums) 
	:param data: input data frame
    :return: preprocessed data
	"""
	latitude = 33.15417 # LATITUDE: 33.15417
	longitude = -83.24056 # LONGITUDE: -83.24056
	tolerance = 1.5

	data2 = data1.loc[(data1['State Name'] == 'Georgia') & (abs (data1['Latitude'] - latitude) < tolerance) \
	& (abs (data1['Longitude'] - longitude) < tolerance)] # data for Georgia
		
	data2.to_csv('data2.csv')

	data2 = pd.read_csv('data2.csv')

	date_time = pd.to_datetime(data2['Date Local'] + ' ' + data2['Time Local'] + " EST") # combining data and time
	data3 = data2[['Sample Measurement']]
	data3['Date_Time'] = date_time
	data3 = data3.rename(columns = {'Sample Measurement' : 'PPM'})

	return data3 # data3 has 2 columns - date time and ppm value


warnings.filterwarnings("ignore")

# Preprocessing hourly SO2 data

data = pd.read_csv('hourly_42401_2015.csv')
target = preprocess(data)
target.to_csv('hourly_SO2_2015.csv')


# Preprocessing hourly O3 data

data = pd.read_csv('hourly_44201_2015.csv')
target = preprocess(data)
target.to_csv('hourly_O3_2015.csv')


# Preprocessing hourly PM data

data = pd.read_csv('hourly_88101_2015.csv')
target = preprocess(data)
target.to_csv('hourly_PM_2015.csv')
