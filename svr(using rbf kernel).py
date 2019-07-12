#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset to perform SVM regression
df = pd.read_csv('Position_Salaries.csv')
X = df.iloc[:, 1:2].values
y = df.iloc[:,2].values

print(X)
print(y)


#feature scaling required as SVR does not do feature scaling automatically
from sklearn.preprocessing import StandardScaler
scaler_X = StandardScaler()
scaler_Y = StandardScaler()
X = scaler_X.fit_transform(X)
Y = scaler_Y.fit_transform(y.reshape(-1,1))
#Y


#importing SVR library from scikit-learn 
from sklearn.svm import SVR
svr = SVR(kernel = 'rbf')

#there are many types of kernel but as the data doesnt look linear, i selected the rbf kernel

svr.fit(X,Y)
y_pred = svr.predict(X)
y_pred


#visualizing the dataset and the curve/line that fits the dataset optimally
plt.scatter(X,Y)
plt.plot(X,svr.predict(X), color = 'red')
plt.title('Support Vector Regression')
plt.xlabel('Independent Variable')
plt.ylabel('predicted values')
plt.show()

#testing the model on some random value
X_test = [[6.5]]
y_pred = scaler_Y.inverse_transform(svr.predict(scaler_X.transform(X_test)))

#o/p = array([252789.13921624])
#as svr.predict would give the value based on the feature scaling performed and to get the actual predicted value the method called
"inverse_tranform" is used

This is how regression is performed on the dataset.




