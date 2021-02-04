#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline
# 
# In this document is the Machine Learning process, where the final data is imported an a machine learning model is created

# ## 1. Import Libraries
# 
# Import needed libraries

# In[ ]:


import sys
import re
import os
import numpy as np
import pandas as pd
import requests
import sqlite3
import pickle


from sqlalchemy import create_engine

import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, fbeta_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.datasets import make_multilabel_classification
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


# ### 2. Load Data
# 
# This part downloads the final data from the ETL Pipeline and creates X and y

# In[ ]:


def load_data(database_filepath):

    '''
    load_data loads the final data that comes from process_data.py and creates X and y DataFrames
    
    INPUT
    database_filepath: Path of database saved by process_data.py
    
    OUTPUT
    X DataFrame with messages
    y DataFrame with categories
     
    '''
    
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Database Project_2', engine)
    
    # drop null columns and fix binary response
    df = df.drop(['child_alone'],axis=1)
    df['related'] = df['related'].map(lambda x: 1 if x == 2 else x)
    
    
    # create X, y
    X = df['message']
    y = df[df.columns[4:]]  
    category_names = y.columns
    
    return X, y, category_names


# ### 3. Text preparation
# 
# This part prepares text for modeling

# In[ ]:


def tokenize(message):
    
    '''
    tokenize gets rid of whitespaces, tokenizes, reduces words to base form and 
    changes letters to lowercase
    
    INPUT
    text variable
    
    OUTPUT
    text variable ready for modeling
    
    '''
    
    message = re.sub(r"[^a-zA-Z0-9]", " ", message)
    
    tokens = word_tokenize(message)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


# ### 4. Model Building
# 
# This part creates a pipeline that builds a machine learning model

# In[ ]:



def build_model():
    
    '''
    build_model defines a pipeline for modelling. There is TfidfTransformer that vectorizes text, CountVectorizer that tokenizes
    and a classifier that defines a model with MultiOutputClassifier
    
    OUTPUT
    pipeline
    
    '''
    
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])

    return pipeline


# ### 5. Model Evaluation
# 
# This part evaluates the model

# In[ ]:


def evaluate_model(model, X_test, y_test, category_names):
    """
    evaluate_model prints metrics from the model
    
    """
    y_pred = model.predict(X_test)
    print(classification_report(y_test.values, y_pred, target_names = category_names))


# ### 6. Save Model
# 
# This part saves the model as pickle

# In[ ]:


def save_model(model, model_filepath):
    """
    save_model saves the pipeline as a pickle
    
    """
    pickle.dump(model, open(model_filepath, "wb"))


# In[ ]:


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '              'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()

