#!/usr/bin/env python
# coding: utf-8

# # ETL Pipeline
# 
# In this document is the ETL process, where the data is extract, transformed and load

# ### 1. Import libraries
# 
# Import needed libraries
# 

# In[ ]:


import requests
import sys
import pandas as pd
import numpy as np
import sqlite3
from sqlalchemy import create_engine


# ### 2. Extract data
# 
# This part loads and merges data

# In[ ]:


def load_data(messages_filepath, categories_filepath):
    
    '''
    load_data loads and merges data from message and categories 
    
    INPUT
    messages_filepath: Path of database containing text features (csv)
    
    categories_filepath: Path of database containing categories features (csv)
    
    OUTPUT
    merged DataFrame
    
    '''

    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories,                         on=['id'])
    return df


# ### 3. Clean Data
# 
# This parte cleans and prepares data to be saved

# In[ ]:


def clean_data(df):
    
    '''    
    clean_data cleans and prepares data to be saved. Splits categories into separate columns
    creates dummy variables and removes duplicates
    
    INPUT
    df: merged df
    
    OUTPUT
    clean and ready to use df
    
    '''
    
    # create categories subset with new column names
    categories = df['categories'].str.split(pat=';', n=-1, expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2])
    categories.columns = category_colnames
    
    
    # transform columns so that each is a dummy variable
    for column in categories:
        categories[column] = categories[column].apply(lambda x:x[-1:])
        
    categories =categories.apply(pd.to_numeric)

    
    # drop original category column and concatenate original DataFrame with categories DataFrame 
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    
    #drop duplicates
    df.drop_duplicates(subset='id', inplace=True)
    
    return df


# ### 4. Load
# 
# This part exports the final transformed DataFrame to SQLite 

# In[ ]:


def save_data(df, database_filename):
    
    '''
    save_data exports the transformed DataFrame to SQL
    
    INPUT
    df
    database_filename
    
    '''
    database_filepath = database_filename
    engine = create_engine('sqlite:///' + database_filepath)
    df.to_sql('Database Project_2', engine, index=False, if_exists='replace')


# In[ ]:


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '              'datasets as the first and second argument respectively, as '              'well as the filepath of the database to save the cleaned data '              'to as the third argument. \n\nExample: python process_data.py '              'disaster_messages.csv disaster_categories.csv '              'DisasterResponse.db')


if __name__ == '__main__':
    main()

