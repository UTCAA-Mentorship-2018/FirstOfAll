# -*- coding: utf-8 -*-
"""
Created on Thu Oct 25 09:11:21 2018

@author: User
"""

import pandas as pd
import matplotlib.pyplot as plt
% matplotlib inline
import math

def Imputing_data(datafile):
# =============================================================================
#   Dropping any duplicated data
# =============================================================================
    data_update = drop_duplications(datafile)
    
# =============================================================================
#   Dividing the entire data into multiple groups and storing in dictionaries
# =============================================================================
    subsets = separate_data(data_update)
    
# =============================================================================
#   Dividing each set of data into Missing and Non-Missing part
#   Obtaining the means of column 'TARGET' for both Missing and Non-Missing part
# ============================================================================= 
    Missing, NonMissing, SeparatedData = dividing_subset_data(subsets)
        
# =============================================================================
#   Getting the indexes for both arrays of Missing and Non-Missing part
# =============================================================================
    MissingIndex, NonMissingIndex, MissingArr, NonMissingArr = getting_indexes(Missing, NonMissing)
    
# =============================================================================
#   Assigning the indexes for imputing the Non-Missing values into the Missing
#   values which the means of 'TARGETS' having the same level of risk    
# =============================================================================
    CorrespondingIndex = mapping_index(MissingIndex, NonMissingIndex, MissingArr, NonMissingArr)
    
# =============================================================================
#   Imputing the Non-Missing Values into Missing values    
# =============================================================================
    UpdatedData = data_imputation(MissingIndex, CorrespondingIndex, SeparatedData)
    
    for i in range(len(UpdatedData)):
        MissingData = UpdatedData[i]['Missing']
        pd.DataFrame(data_update).update(MissingData)
        
# =============================================================================
#   Checking any missing data left in the dataset
# =============================================================================
    
    if data_update.isnull().sum().any() != 0:
        return 'Oops! There are still missing data...'
    
    else:
        return data_update

def drop_duplications(data):
# =============================================================================
    '''
    Trying to catch any duplications of the data in every row:
        1. Checking the duplications
        2. Dropping the duplicated one, only keep the first one left
        3. Reseting the index after removing duplications
    '''
# =============================================================================
    
    if data.duplicated().any() == True:
        print("Deleting the duplicated rows...")
        data_drop = data.drop_duplicates(keep = 'first')
        print("Reseting the rows of data...")
        data_reset = data_drop.reset_index(drop=True)
        return data_reset
    
    else:
        print('There are no any duplications in the dataset...')
        return data

# =============================================================================
#   Rounding the number to the closest hundreds
# =============================================================================
def round_up(n, decimals = 0):
    
    multiplier = 10 ** decimals
    
    return math.ceil(n * multiplier) / multiplier

def separate_data(data):
# =============================================================================
    '''
        Partitioning the entire data by:
            1. Rounding the observations to the closest hundreds for easier grouping data
            2. Setting each group has 11000 observations
            3. Storing each sub-dataset into a dictionary
            4. Forming 28 groups of sub-datasets and dictionaries in total
            5. Storing all the dictionaries into a single dictionary called SubSets
    '''
# =============================================================================
    SubSets = dict()                            # To save all the sub datasets
    start = 0                                   
    stop = int(round_up(data.shape[0], -3))     # Rounding the length of observations up to nearest hundreds
    Length = 11000                              # Defining the size of each subset
    nums = int((stop - start) / Length)         # Defining how many subsets that the observation can be devided
    
    i = 0
    print('Forming the subsets...')
    while(i < stop):
        for sets in range(int(nums)):
            # Partitioning every 11000 observation as a sub dataset
            SubSets[sets] = pd.DataFrame(data.iloc[i:i+Length])
            i += Length
            print('The subset ' + str(sets) + ' has been formed with length of ' + str(len(SubSets[sets])))
    
    return SubSets

def dividing_subset_data(SubSet):
# =============================================================================
    '''
        For each Sub-DataSet:
        1. The observations with Missing and Non-Missing data are separated into two groups
        2. The TARGET mean for both Missing and Non-Missing data are saved in two separated arrays
        
        In the end, there will be 28 values in each array which has the same length as
        the SubSet dictionary
    '''    
# =============================================================================
    Length = len(SubSet)
    Target_Means_Missing = []                   # To save all the TARGET means with Missing observations
    Target_Means_Non_Missing = []               # To save all the TARGET means with Non-Missing observations
    Separated_Data = {}                         # To save the Missing and Non-Missing data for each subset
    
    print('Separating each subset into Missing and Non-Missing...\n')
    for sets in range(Length):
        length = len(SubSet[sets])
        # Defining dictionaries in each subset to separate data into Missing and Non-Missing
        Separated_Data[sets] = {'Missing': [], 'Non-Missing': []}
        # Assigning data to the corresponding arrays
        for rows in range(length):
            if SubSet[sets].iloc[rows].isnull().sum() != 0:
                Miss = Separated_Data[sets]['Missing']
                Miss.append(SubSet[sets].iloc[rows])
            else:
                NonMiss = Separated_Data[sets]['Non-Missing']
                NonMiss.append(SubSet[sets].iloc[rows])
        # Saving TARGET mean with Missing observations to a new array
        Missing = pd.DataFrame(Miss)
        Missing_Means = Missing['TARGET'].mean()
        Target_Means_Missing.append(Missing_Means)
        print('The TARGET Means of subset ' + str(sets) + ' with Missing observations is ' + str(Missing_Means))
        
        # Saving TARGET mean with Non-Missing observations to a new array
        NonMissing = pd.DataFrame(NonMiss)
        NonMissing_Means = NonMissing['TARGET'].mean()
        Target_Means_Non_Missing.append(NonMissing_Means)
        print('The TARGET Means of subset ' + str(sets) + ' with Non-Missing observations is ' + str(NonMissing_Means))
            
    return Target_Means_Missing, Target_Means_Non_Missing, Separated_Data

def getting_indexes(Target_Miss, Target_NonMiss):
# =============================================================================
    '''
        To get the indexes for both arrays:
        1. Concatinating two arrays as a table
        2. Getting the index of TARGET mean with Missing data
        3. Catching the index of TARGET mean with Non-Missing data where the mean is greater than 0.07
           (analyzed from the graph)
    '''
# =============================================================================
    
    TM = pd.Series(Target_Miss)
    TNM = pd.Series(Target_NonMiss)
    
    # Concatinating the TARGET means with and without Missing observations and rounding up to 4 decimal digits
    Target_Means_Table = pd.concat([TM, TNM], axis = 1)
    Target_Means_Table_Renames = Target_Means_Table.rename(columns = {0: 'TARGET Means with Missing', 1: 'TARGET Means with Non-Missing'})
    Target_Means_Table_Renames = Target_Means_Table_Renames.round(4)
    TABLES = pd.DataFrame(Target_Means_Table_Renames)
    print('There are {} subsets of TARGET Means'.format(Target_Means_Table_Renames.shape[0]) + '\n')
    print(TABLES)
    
    # Separating the columns of the TARGET means table
    Target_Missing = Target_Means_Table_Renames['TARGET Means with Missing']
    Target_NonMissing = Target_Means_Table_Renames['TARGET Means with Non-Missing']
    
    # Scatter plotting the distribution of TARGET means for Missing and Non-Missing
    plt.figure()
    plt.plot(Target_Missing, 'bo')
    plt.plot(Target_NonMissing, 'r*')
    plt.title('Distributions of TARGET Means')
    plt.xlabel('subset indexes')
    plt.ylabel('TARGET Means')
    
    # Catching the TARGET means with Non-Missing observations greater than 0.07
    Sub_Target_NonMissing = Target_NonMissing[Target_NonMissing > 0.07]
    
    # Obtaining the indexes for both Missing and sub-Non-Missing observations
    Target_Miss_Index = Target_Missing.index
    Target_NonMiss_Index = Sub_Target_NonMissing.index
    
    return Target_Miss_Index, Target_NonMiss_Index, Target_Missing, Sub_Target_NonMissing

def mapping_index(Miss_Index, NonMiss_Index, Missing_Val, Sub_NonMissing_Val):
# =============================================================================
    '''
        To get the corresponding Non-Missing Data index:
        1. Finding the difference between each element for TARGET mean (in Missing data)
           and TARGET mean greater than 0.07 (in Non-Missing data).  
        2. Finding the minimum values in each set of differences from the diff dictionary
        3. Finding the indexes where the minimum difference occurs in the set of TARGET mean greater than 0.07
        4. Mapping the indexes obtained last step to get the corresponding index from the TARGET Mean Table
    '''
# =============================================================================
    Diff = dict()       # To save all the differences between the TARGET means 
                        # with and without Missing   
    
    DiffMin = dict()    # To save the minimum values for each set of difference
    
    Indexes = []        # To save the corresponding indexes in the sub-Non-Missing 
                        # dataset for the minimum difference 

    for i in Miss_Index:
        Diff[i] = []
        DiffMin[i] = []
        
        # Finding the differences of TARGET means between Missing and sub-Non-Missing observations
        for j in NonMiss_Index:
            Diff[i].append(abs(Missing_Val.loc[i] - Sub_NonMissing_Val.loc[j]))
        
        # Collecting all the minimum differences among each set of differences
        DiffMin[i].append(min(Diff[i]))
        # Collecting the indexes in the sub-Non-Missing for set of minimum differences  
        Indexes.append(Diff[i].index(DiffMin[i]))
        # Obtaining the indexes from the TARGET means Table
        Mapping_Index = NonMiss_Index[Indexes]
    
    print('The indexes of minimum differences occurring in the TARGET means with Non-Missing observations are: ')
    print(Indexes, '\n')
    return Mapping_Index

def data_imputation(Miss_Index, NonMiss_Index, Separated_Data):
# =============================================================================
    '''
        To impute the missing data in the dataset:
        1. Using the data from Non-Missing Sub-DataSets to fill in the Data with NaN
           based on the indexes obtained from last step
        2. For the categorical missing data, the most frequent values in the columns 
           of the corresponding Non-Missing Sub-DataSets are filled into the Missing Sets
        3. For the types of 'float64' and 'int64', if the length of the distinct values
           are less than 50, the same method is applied as the categorical variables
        4. For other values, the means in the columns of the corresponding Non-Missing 
           Sub-DataSets are filled into the Missing Sets
        5. Updating the dataset with the new fill in data
    '''
# =============================================================================
    i = 0
    j = 0
    MissLength = len(Miss_Index)
    NonMissLength = len(NonMiss_Index)
    
    while((i < MissLength) and (j < NonMissLength)):
            
        # Separating the Missing and Non-Missing observations according to the mapping_index function
        Separated_Missing = Separated_Data[i]['Missing']
        Separated_NonMissing = Separated_Data[j]['Non-Missing']
            
        Miss_Part = pd.DataFrame(Separated_Missing)
        NonMiss_Part = pd.DataFrame(Separated_NonMissing)
        Columns = Miss_Part.columns.values
        Length = len(Columns)

        for item in range(Length):
                
            # Obtaining the column variables
            MC = Columns[item]
                
            # Imputing the missing categorical data with the most frequent values in the columns of corresponding Non-Missing observations
            if Miss_Part[MC].dtypes == 'object':
                MostFrequent = NonMiss_Part[MC].value_counts().index[0]
                Miss_Part[MC] = Miss_Part[MC].fillna(MostFrequent)
                
            # Imputing the missing data with the median values in the columns of corresponding Non-Missing observations where the length of distinctive values are less than 50 and the column data types are 'int64' or 'float64'
            elif (len(Miss_Part[MC].value_counts()) <= 50) and ((Miss_Part[MC].dtypes == 'float64') or (Miss_Part[MC].dtypes == 'int64')):
                Median = NonMiss_Part[MC].median()
                Miss_Part[MC] = Miss_Part[MC].fillna(Median)
                
            # Imputing other missing data with the mean values in the columns of corresponding Non-Missing observations
            else:
                mean = NonMiss_Part[MC].mean()
                Miss_Part[MC] = Miss_Part[MC].fillna(mean)
                
        print('The Missing of separated dataset ' + str(Miss_Index[i]) + ' has been imputed with the Non-Missing of separated dataset ' + str(NonMiss_Index[j]))
            
        # Updating the observations in the separated datasets
        Separated_Data[i]['Missing'] = Miss_Part
        Separated_Data[j]['Non-Missing'] = NonMiss_Part
        
        i += 1
        j += 1
    
    return Separated_Data
