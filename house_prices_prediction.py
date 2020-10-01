#Using Regression

#Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing the dataset
dataset =  pd.read_csv("housing.csv")
X = dataset.iloc[:, [0,1,2,3,4,5,6,7,9]].values
y = dataset.iloc[:, [8]].values

#Managing of the missing values of the dataset
from sklearn.preprocessing import Imputer 
imputer= Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(X[:,[4]]) 
X[:,[4]]=imputer.transform(X[:,[4]])

# Encoding categorical data of Ocean Proximity
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder , LabelEncoder
labelencoder_X = LabelEncoder()
X[:,8] = labelencoder_X.fit_transform(X[:,8])
onehotencoder = OneHotEncoder(categorical_features = [8])
X = onehotencoder.fit_transform(X).toarray()


# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
X_train=sc_x.fit_transform(X_train)
X_test=sc_x.transform(X_test)

# Training the Multiple Linear Regression model on the Training set
from sklearn.linear_model import LinearRegression
lin_regressor = LinearRegression()
lin_regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = lin_regressor.predict(X_test)
#Analysing the results error
from sklearn.metrics import mean_squared_error
housing_predictions = lin_regressor.predict(X_test)
lin_mse = mean_squared_error(y_test, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print(lin_rmse)

#Training the Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
tree_reg = DecisionTreeRegressor()
tree_reg.fit(X_train, y_train)

#Testing and analysising  the results' error
housing_predictions = tree_reg.predict(X_test)
tree_mse = mean_squared_error(y_test, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print(tree_rmse)

#Trying the Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

housing_predictions = forest_reg.predict(X_test)
forest_mse = mean_squared_error(y_test, housing_predictions)
forest_rmse = np.sqrt(tree_mse)
print(forest_rmse)