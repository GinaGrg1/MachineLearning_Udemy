
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33, random_state = 0)

# Feature Scaling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)'''

# Fitting Simple Linear Regression to the Training Set.
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Needs fitting before predicting. Predicting the Test set results.
# Compare this with y_test which are the real values.
y_pred = regressor.predict(X_test)

# Visualising the Training set results.
plt.scatter(X_train, y_train, color = 'red')
# plot the predicted training test.
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #regression/predicted line
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.sho()

# Next we need to plot the test sets against the blue (regression) line.
plt.scatter(X_test, y_test, color = 'red')
# plot the predicted training test.
plt.plot(X_train, regressor.predict(X_train), color = 'blue') #regression/predicted line
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of experience')
plt.ylabel('Salary')
plt.sho()


