import pandas as pd

def miss_values_table(df):
    miss_values = df.isnull().sum()
    miss_values_percent = 100 * df.isnull().mean()
    
    miss_values_table = pd.concat([miss_values, miss_values_percent], axis = 1)
    miss_values_table_rename_columns = miss_values_table.rename(columns = {0: 'Missing Values', 1: '% of Total Values'})
    miss_values_table_rename_columns = miss_values_table_rename_columns[miss_values_table_rename_columns.iloc[:, 1] != 0].sort_values('% of Total Values', ascending = False).round(1)
    
    print("Your selected dataframe has " + str(df.shape[1]) + "columns.\n"
         "There are " + str(miss_values_table_rename_columns.shape[0]) + " columns that have missing values.")
    
    return miss_values_table_rename_columns
