#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 22:12:50 2017

@author: ReginaGurung
"""
# Multiple Linear Regression

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

os.chdir(os.getcwd()+'/Documents/MachineLearning_Udemy/Lesson_1/')

# Importing the dataset
dataset = pd.read_csv('Lesson_1/data/50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, len(dataset.columns)-1].values

# Encoding categorical data. Encoding the independent variable.
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 3] = labelencoder_X.fit_transform(X[:, 3])
# Creates two dummy variables off 'state'.
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

# Avoiding Dummy variable trap. There are 3 DVs and we will remove 1.
# This is done by the library and is not needed to do manually.
X = X[:, 1:] 

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature scaling is not required as its also done by the library.
# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''