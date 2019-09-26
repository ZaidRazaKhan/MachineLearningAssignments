import pandas as pd
import numpy as np
import math


def read_file(file_path):
    data = pd.read_csv(file_path)
    return data

def conversion_from_string_to_date_time(data_frame, date_time_string_column):
    data_frame[date_time_string_column] = pd.to_datetime(data_frame[date_time_string_column])
    return data_frame

def get_columns_name(data_frame):
    columns_name = []
    for i in data_frame.columns:
        columns_name.append(i)
    return columns_name


def constant_interpolation(data_frame, column_name, rounding_factor):
    """
    Takes a data_frame with column name and returns a sum, mean, countOfNumbers, old column, and new column of "column_name"
    """
    sum = 0
    count = 0
    mean = 0
    old_column = data_frame[column_name]
    for data in old_column:
        if not math.isnan(data):
            sum += data
            count +=1
    if old_column.shape[0] == count:
        return sum, count, mean, old_column, old_column 
    mean = sum/count
    new_column = []
    for data in old_column:
        if math.isnan(data):
            new_column.append(round(mean, rounding_factor))
        else:
            new_column.append(data)
    return sum, count, mean, old_column, new_column



def categorical_to_numeric(data_frame, column_name):
    new_data_frame = pd.get_dummies(data_frame[column_name])
    return new_data_frame



def join_data(data_frame1, data_frame2):
    joined_data_frame = data_frame1.join(data_frame2)
    return joined_data_frame



def averaging_rows(data_frame, categorical_data_list):
    m = data_frame.shape[0]
    new_df = pd.DataFrame(columns = data_frame.columns)
    columns = data_frame.columns
    for i in range(len(columns)):
        column = columns[i]
        col = []
        if column in categorical_data_list:
            i = 0
            while i in range(0,m):
                col.append(data_frame.iloc[i][column])
                i += 3
            new_df[column] = col
            continue
        col = []
        i = 0
        while i in range(0,m):
            val1 = data_frame.iloc[i][column]
            if i+1 < m:
                val2 = data_frame.iloc[i+1][column]
            else:
                val2 = 0
            if i+2 < m:	
                val3 = data_frame.iloc[i+2][column]
            else:
                val3 = 0
            avg = (val1+val2+val3)/3
            col.append(avg)
            i += 3
        new_df[column] = col
    return new_df


def save_data_frame(data_frame, file_name):
    data_frame.to_csv(file_name)
    print('successfully saved file: '+ file_name)


def natural_join_data_frames(data_frame1, data_frame2, common_column):
    natural_joined_data_frame = pd.merge(data_frame1, data_frame2, on = common_column, how = 'inner')
    return natural_joined_data_frame


# def conversion_from_string_to_date_time(data_frame, column_name):
#     return data_frame[column_name].astype('datetime64[ns]')

def remove_all_null_entries(data_frame):
    for column in data_frame.columns:
        data_frame = data_frame[pd.notnull(data_frame[column])]
    return data_frame