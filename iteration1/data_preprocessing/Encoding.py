# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:45:11 2018

@author: User
"""

# Pandas
import pandas as pd

# Encoding
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Spliting data
from sklearn.model_selection import train_test_split

def Encoding_Data(data):
    df = data.copy()
    labelcount = 0

    for column in df:
        if df[column].dtypes == 'object':
            if len(df[column].unique()) <= 2:
                print('The column of {} is encoding...'.format(column))
                TBEncode = df[column]
                labelencoder = LabelEncoder()
                LabelEncoded = labelencoder.fit_transform(TBEncode)
                df[column] = LabelEncoded

            else:
                print('The column of {} is encoding...'.format(column))
                df = pd.get_dummies(df, columns = [column])

            labelcount += 1

    print('%d columns were label encoded' % labelcount)
    return df

def scale_split(data, target, test_pro1, test_pro2):
    
    scaled = {}
    
    data_train, data_test, target_train, target_test = train_test_split(data, target, test_size = test_pro1, shuffle = True)
    data_train, data_val, target_train, target_val = train_test_split(data_train, target_train, test_size = test_pro2, shuffle = True)
    
    data_scaled = StandardScaler()
    data_scaled.fit(data_train)
    
    data_train_scaled = data_scaled.transform(data_train)
    data_test_scaled = data_scaled.transform(data_test)
    data_val_scaled = data_scaled.transform(data_val)
    
    return data_train_scaled, data_test_scaled, data_val_scaled, target_train, target_test, target_val
