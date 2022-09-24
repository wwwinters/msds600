#!/usr/bin/env python

"""
########################################################################################################
#
# Wiley Winters
# Assignment Week 5 Python File
# September 26, 2022
#
# Create a python script to load churn data into a pandas dataframe, load a previously created model,
# use the model to predict churn, and print results
# 
# Dependencies -- the saved model has to be in the same directory as this script.
#
# Version 1.0
#
########################################################################################################
"""

# 
# Load required packages and funtions
#
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier

#
# I am a Linux person so I like to use arguments to pass in required parameters or
# change the behavior of the script.
#
# It is also possible to provide functionality to read parameters from a configuration file.
# In this case, I believe the use of command line arguments is sufficient.
#
parser = argparse.ArgumentParser()
parser.add_argument('--datafile', default='../data/week5/new_churn_data.csv',
                    help='Path to the csv file containing the churn data.')
parser.add_argument('--bestmodel', default='xgbc',
                    help='Name of the saved tpot model.  Choices are etc and xgbc')
args = parser.parse_args()
data_file = args.datafile
best_model = args.bestmodel

#
# The training data will remain constant for the time being; therefore, will
# hard-code it.
#
train_file = '../data/prepped_churn_data.csv'

#
# Use pandas to read in the csv file into a dataframe and break out train and test
# data.
#
def load_data(train_file, data_file):
    train_data = pd.read_csv(train_file, index_col='customerID')
    new_data = pd.read_csv(data_file, index_col='customerID')
    return train_data, new_data
#
# Use xgbc model to make predictions on the data in the dataframe
# Average CV score on the training set was: 0.8000725681603165
#
def xgbc_predictions(train_data, new_data):
    xgbc = XGBClassifier(learning_mode=0.1, max_depth=2, min_child_weight=2,
                         n_estimators=100, n_jobs=-1, subsample=0.45, verbosity=0)
    features = train_data.drop('Churn', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, train_data['Churn'],
                                                        random_state=42)
    xgbc.fit(X_train, y_train)
    new_data = new_data.drop('charge_per_tenure', axis=1)
    predictions = xgbc.predict(new_data)
    return predictions

#
# Use etc model to make predictions on data in the dataframe
# Average CV score on the training set was: 0.8402497539950419
#
def etc_predictions(train_data, new_data):
    etc = ExtraTreesClassifier(bootstrap=True, criterion="entropy", max_features=0.8, 
                               min_samples_leaf=19, min_samples_split=5, n_estimators=100)
    features = train_data.drop('Churn', axis=1)
    X_train, X_test, y_train, y_test = train_test_split(features, train_data['Churn'],
                                                        random_state=42)
    etc.fit(X_train, y_train)
    new_data = new_data.drop('charge_per_tenure', axis=1)
    predictions = etc.predict(new_data)
    return predictions
 
#
# Pull it all together in the main function
#
def main():
    train_data, new_data = load_data(train_file, data_file)
    if best_model == 'xgbc':
        predictions = xgbc_predictions(train_data, new_data)
    elif best_model == 'etc':
        predictions = etc_predictions(train_data, new_data)
    else:
        print('Valid bestmodel values are: etc and xgbc')
    print(predictions)

if __name__ == '__main__':
    main()