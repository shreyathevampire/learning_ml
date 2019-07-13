#importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Social_Network_Ads.csv')

#taking only age and estimated salary into consideration
#X = col 2 and 3
X = df.iloc[:, 2:4].values
Y = df.iloc[:, -1].values


#feature scaling required so that age and salary both come to the same scale
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(X)


from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,Y, test_size = 0.25, random_state = 0)

#using logistic regression to perform Classification

#1. import library
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)

#2.Fitting the classifier
classifier.fit(Xtrain,ytrain)

#3. Predicting the test values
y_pred = classifier.predict(Xtest)

#4. calculating the score 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)
print(cm)

#op: 
Out[25]:
array([[68,  0],
       [32,  0]], dtype=int64)
       
       
classifier.score(Xtrain, ytrain)


#o/p: 0.63

classifier.score(Xtest, ytest)

#o/p: 0.68

