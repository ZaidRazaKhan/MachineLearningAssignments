from DataLoader import DataLoader
import numpy
import pandas as pd

class DataManipulator():
    
    # Loader object for handling file operations
    loader = DataLoader()
    
    # Weather Condition Codes
    weatherConditions = {'Clear':0, 'Mist':1, 'Partly Cloudy':2, 'Overcast':3, 'Mostly Cloudy':4,
       'Light Rain':5, 'Thunderstorm':6, 'Heavy Rain/Thunderstorm':7, 'Rain':8,
       'Heavy Rain':9, 'Fog':10, 'Light Rain/Thunderstorm':11, 'Thunder':12, '':13, 'N/A':14, numpy.nan:numpy.nan,
       'Drizzle':16, 'Light Drizzle':17, 'Unknown Precip':18, 'Ice Fog':19,
       'Heavy Drizzle':20, 'Haze':21, 'Light Snow':22, 'Snow':23, 'Heavy Snow':24, 'Mostly Clear':25}

    # Wind Direction Codes
    windDirections = {'N':0, 'E':1, 'W':2, 'S':3, 'NE':4, 'NW':5, 'NNE':6, 'NNW':7, 'ENE':8,
        'ESE':9, 'WNW':10, 'WSW':11, 'SE':12, 'SW':13, 'SSE':14, 'SSW':15, '':numpy.nan}

    # Convert to categorial
    def toCategorial(self, data, column, categories):
        for key, value in categories.items():
            data.insert(13, key, list(map(int, (data[column]==value).tolist())))
        return data

    # Remove Unnecessary Data from other locations
    def keepNecessary(self, folders, columnName, cellData):
        count = 1
        for folder in folders:
            print('Removing data from files of folder: ' + folder)
            files = self.loader.filesInFolderWithPattern('ML Dataset/'+folder, '*.csv')
            for file in files:
                print('['+str(count)+'] '+file+' .... ', end='', flush=True)
                data = self.loader.loadCSVFile(file)
                data = data[data[columnName] == cellData]
                data.to_csv(file, index=False)
                print('done')
                count += 1

    # Remove Columns of all files in the folder
    def keepColumns(self, folders, colnames):
        count = 1
        for folder in folders:
            print('Removing columns from files of folder: ' + folder)
            files = self.loader.filesInFolderWithPattern('ML Dataset/'+folder, '*.csv')
            for file in files:
                print('['+str(count)+'] '+file+' .... ', end='', flush=True)
                data = self.loader.loadCSVFile(file)
                data = data[colnames]
                data.to_csv(file, index=False)
                print('done')
                count += 1
        return

    # Reduce Redundant Data
    def averageData(self, data):
        l = len(data.index)
        meanCols = [1, 2, 3, 4, 5, 6, 7, 8, 10]
        modeCols = [9, 11]
        dropList = []
        i = 0
        while i < l:
            # print("\r"+str(i)+"/"+str(l), end="", flush=True)
            k = i
            while (k+1 < l) and (data.at[i, 'Date_Time'][11:13] == data.at[k+1, 'Date_Time'][11:13]):
                k += 1
            if k+1 == l: data.iloc[i, meanCols] = data.iloc[i:k, meanCols].mean()
            else: data.iloc[i, meanCols] = data.iloc[i:k+1, meanCols].mean()
            if k+1 == l: data.iloc[i, modeCols] = data.iloc[i:k+1, modeCols].mode().iloc[0, [0, 1]]
            else: data.iloc[i, modeCols] = data.iloc[i:k+1, modeCols].mode().iloc[0, [0, 1]]
            if k+1 < l: dropList.extend(range(i+1, k+1))
            else: dropList.extend(range(i+1, k))
            i = k+1
        data.drop(dropList, inplace=True)
        return data

    def averagePollutant(self, data):
        l = len(data.index)
        dropList = []
        i = 0
        while i < l:
            k = i
            while (k+1 < l) and (data.at[i, 'Time Local'] == data.at[k+1, 'Time Local']):
                k += 1
            data.at[i, 'Sample Measurement'] = data.iloc[i:k+1, 4].mean()
            if k+1 < l: dropList.extend(range(i+1, k+1))
            else: dropList.extend(range(i+1, k))
            i = k+1
        data.drop(dropList, inplace=True)
        return data

    # Normalize columns
    def normalize(self, data):
        cols = ['air_temp_set_1', 'relative_humidity_set_1', 'wind_speed_set_1', 'wind_direction_set_1', 'wind_gust_set_1', 
        'precip_accum_one_hour_set_1', 'visibility_set_1', 'dew_point_temperature_set_1d', 'pressure_set_1d', 'Sample Measurement']
        for col in cols:
            data[col] = (data[col]-data[col].mean())/data[col].std()
        return data

    # Preprocess Data
    def preprocess(self):
        pollutantFolders = ['O3', 'SO2', 'PM2_5_Non_FRM']
        meteorologicalDataFolders = ['KIGQ', 'KLOT']
        colnames = ['Date Local', 'Time Local', 'Date GMT', 'Time GMT', 'Sample Measurement']
        inputCols = ['Date_Time', 'air_temp_set_1', 'relative_humidity_set_1', 'wind_speed_set_1', 'wind_direction_set_1', 'wind_gust_set_1', 
        'precip_accum_one_hour_set_1', 'visibility_set_1', 'dew_point_temperature_set_1d', 'wind_cardinal_direction_set_1d',
        'pressure_set_1d', 'weather_condition_set_1d']
        self.keepNecessary(pollutantFolders, 'County Name', 'Cook')
        self.keepColumns(pollutantFolders, colnames)
        self.keepColumns(meteorologicalDataFolders, inputCols)
        data = self.loader.loadFilesInFolderWithPattern("ML Dataset/KIGQ", "*.csv")
        data = data.interpolate(limit_direction='both')
        data.wind_cardinal_direction_set_1d = data.wind_cardinal_direction_set_1d.interpolate(method='pad')
        data.weather_condition_set_1d = data.weather_condition_set_1d.interpolate(method='pad')
        data.wind_cardinal_direction_set_1d = data.wind_cardinal_direction_set_1d.apply(lambda x: self.windDirections[x])
        data.weather_condition_set_1d = data.weather_condition_set_1d.apply(lambda x: self.weatherConditions[x])

        SO2_data = self.loader.loadCSVFile("SO2_data.csv")
        O3_data = self.loader.loadCSVFile("O3_data.csv")
        PM2_5_data = self.loader.loadCSVFile("PM2_5_data.csv")
        input_data = self.loader.loadCSVFile("input.csv")

        # Get common rows of input and output
        SO2_data = pd.merge(input_data, SO2_data, how='inner', on=['Date_Time'])
        PM2_5_data = pd.merge(input_data, PM2_5_data, how='inner', on=['Date_Time'])
        O3_data = pd.merge(input_data, O3_data, how='inner', on=['Date_Time'])

        # Categorial
        self.toCategorial(SO2_data, 'wind_cardinal_direction_set_1d', self.windDirections)
        self.toCategorial(SO2_data, 'weather_condition_set_1d', self.weatherConditions)

        # Normalize columns of data
        SO2_data  = self.normalize(SO2_data)
        O3_data = self.normalize(O3_data)
        PM2_5_data = self.normalize(PM2_5_data)

        # Test Train Split
        SO2_train = SO2_data.loc[:80000, :]
        SO2_test = SO2_data.loc[80000:, :]
        O3_train = O3_data.loc[:80000, :]
        O3_test = O3_data.loc[80000:, :]
        PM2_5_train = PM2_5_data.loc[:80000, :]
        PM2_5_test = PM2_5_data.loc[80000:, :]

        # Save CSVs for training
        SO2_train.to_csv("SO2_train.csv", index=False)
        SO2_test.to_csv("SO2_test.csv", index=False)
        O3_train.to_csv("O3_train.csv", index=False)
        O3_test.to_csv("O3_test.csv", index=False)
        PM2_5_train.to_csv("PM2.5_train.csv", index=False)
        PM2_5_test.to_csv("PM2.5_test.csv", index=False)
        return