# -*- coding: utf-8 -*-
"""
Created on Sun Nov 25 22:45:11 2018

@author: User
"""

# Pandas
import pandas as pd

# Encoding
from sklearn.preprocessing import LabelEncoder

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