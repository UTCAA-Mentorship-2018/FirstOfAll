# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:11:21 2018
@author: User
"""

import pandas as pd

datafile = "C:/Projects/exercises/DataScience/application_train.csv"
def data_preprocess(datafile):
    '''Reading the data from the file'''
    print("Loading the data...")
    data = pd.read_csv(datafile)
    df = pd.DataFrame(data)
    
    '''
    Trying to catch any duplications of the data in every row:
        1. Checking the duplications
        2. Dropping the duplicated one, only keep the first one left
        3. Reseting the index after removing duplications
    '''
    if df.duplicated().all() == True:
        print("Deleting the duplicated rows...")
        df.drop_duplicates(keep = 'first')
        print("Reseting the rows of data...")
        df.reset_index(drop=True)
    
    '''Dealing with the missing data (NAN):
        1. Collect all the columns with the NAN data
        2. For non-numeric data, if the columns contain any NANs, the entire row will be dropped because the data 
           cannot be predicted
        3. For the numeric data, if the columns contain any NANs, the mean of the entire column will be replaced
           with NANs
    '''
    column = df.columns
    withNAs = []
    print("Collecting the columns with NAs...")
    for col in column:
        NAs = df[col].isnull().sum()
        if NAs != 0:
            withNAs.append(col)
    
    print("Dropping and filling in the missing data...")
    for item in withNAs:
        if df[item].dtype == 'object':
            df = df[pd.notnull(df[item])]
        else:
            mean = df[item].mean()
            df[item] = df[item].fillna(mean)
    '''Update the original csv file'''
    print("Updating the csv file...")
    return df.to_csv(datafile)

data_preprocess(datafile)
