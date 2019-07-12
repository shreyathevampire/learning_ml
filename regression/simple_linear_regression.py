import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('Salary_Data.csv')
df.head()

#change the datatype of salary columns from float64 to int64 - pandas.to_numeric() method is used
df['Salary']  = pd.to_numeric(df['Salary'],downcast = 'signed')
#df
X = df.iloc[:, : -1]
Y = df.iloc[:, -1]

from sklearn.model_selection import train_test_split as tt
X_train,X_test,ytrain,ytest = tt(X,Y,test_size = 1/3, random_state = 0)
#X_train
#ytrain

#feature scaling is not required here as the linear regression model takes care of it
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

# here the model learns the correlation between the independent and dependent variables and helps to predict values later
regressor.fit(X_train,ytrain)

#predicting the test dataset
y_pred = regressor.predict(X_test)

#visualize the training set results
plt.scatter(X_train, ytrain, color = 'red')
plt.scatter(X_train, ytrain, color = 'red')
plt.plot(X_train, regressor.predict(X_train),color = 'blue')
plt.title('sal vs exp(Training_Set)')
plt.xlabel('EXP')
plt.ylabel('SAL')
plt.show()

#taking an array of one sample for predicting the result
X = [6]
Y = np.array(X)
print(regressor.predict(Y.reshape(1,-1)))


plt.scatter(X_test, ytest, color = 'red')
plt.plot(X_test, y_pred, color= 'blue')
plt.title('Exp vs sal (Test-Set)')
plt.show()

