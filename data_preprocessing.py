#importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#importing dataset and storing it in a form of a dataframe using pandas
dataset = pd.read_csv('Data.csv')
#taking feature variables
X = dataset.iloc[:, :-1].values
#taking labels in y 
y = dataset.iloc[:, 3].values

#Imputer is used to replace missing data with the mean of the column values
from sklearn.preprocessing import Imputer
#OneHotEncoder is used to replace categorical data with numbers and create dummy variables
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
imputer = Imputer(missing_values = 'NaN', axis = 0, strategy = 'mean')
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:, 1:3])
print(X)
labelencoder = LabelEncoder()
X[:,0] = labelencoder.fit_transform(X[: , 0])
onehotencoder  = OneHotEncoder(categorical_features = [0])
X = onehotencoder.fit_transform(X).toarray()
print(X)
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)
print(y)

#splitting data into training and testing data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.2,random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

