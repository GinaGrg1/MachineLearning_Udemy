
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

# Fitting Multiple Linear Regression to the training set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train) # simliar to doing lm(y ~ x) in R

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Building the optimal model using Backward Elimination.
import statsmodels.formula.api as sm
# create a new column with 1's. This new col corresponds to X0 term or the intercept.
X = np.append(arr = np.ones((50, 1)).astype(int), # 50 rows and 1 column.
              values = X,
              axis = 1) 

# X optimal will only consists of predictors which are statistically significant.
X_Optimal = X[:, [0, 1, 2, 3, 4, 5]] # indices of the predictors.
regressor_OLS = sm.OLS(endog = y, exog = X_Optimal).fit() #Ordinary Least Squares. Stp 2 of BE.
regressor_OLS.summary()
# Here x2 has the highest p-value so we remove this variable from X_Optimal.
X_Optimal = X[:, [0, 1, 3, 4, 5]] # indices of the predictors.
regressor_OLS = sm.OLS(endog = y, exog = X_Optimal).fit() #Ordinary Least Squares. Stp 2 of BE.
regressor_OLS.summary()
# Here x1 has the highest p-value
X_Optimal = X[:, [0, 3, 4, 5]] # indices of the predictors.
regressor_OLS = sm.OLS(endog = y, exog = X_Optimal).fit() #Ordinary Least Squares. Stp 2 of BE.
regressor_OLS.summary()
X_Optimal = X[:, [0, 3, 5]] # indices of the predictors.
regressor_OLS = sm.OLS(endog = y, exog = X_Optimal).fit() #Ordinary Least Squares. Stp 2 of BE.
regressor_OLS.summary()
# The highest p-value now is 0.06 which is still higher than 0.05.
X_Optimal = X[:, [0, 3]] # indices of the predictors.
regressor_OLS = sm.OLS(endog = y, exog = X_Optimal).fit() #Ordinary Least Squares. Stp 2 of BE.
regressor_OLS.summary()

