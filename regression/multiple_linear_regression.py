import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


df = pd.read_csv('50_Startups.csv')
df.head(10)
df['R&D Spend']


#convert float to int 
df['R&D Spend'] = df['R&D Spend'].astype(np.int64)
df['Administration'] = df['Administration'].astype(np.int64)
df['Marketing Spend'] = df['Marketing Spend'].astype(np.int64)
df['Profit'] = df['Profit'].astype(np.int64)

df.head()

X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# we need to encode the State as state is a categorical feature
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder  = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [3])

#print(X[:,3])

X[: , 3] = labelencoder.fit_transform(X[:, 3])
X = onehotencoder.fit_transform(X).toarray()

#avoiding the dummy variable trap by eliminating one of the dummy variables as it helps in improving the model and avoids unncessary noise in data
X = X[:,1:]


from sklearn.model_selection import train_test_split as tt
xtrain, xtest, ytrain, ytest = tt(X,Y,test_size = 0.2, random_state = 0)
#xtrain.shape


#dataset is preprocessed successfully, now its time to fit the training data 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(xtrain,ytrain)

#predicting the test data
y_pred = regressor.predict(xtest)

#y_pred

