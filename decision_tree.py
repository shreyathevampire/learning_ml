#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

importing the dataset
df = pd.read_csv('Position_Salaries.csv')

X = df.iloc[:, 1:2].values
Y = df.iloc[:, 2].values


#importing the Decision Tree Regressor
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state = 0)

#fitting the regressor on the data-set
regressor.fit(X,Y)


#visualizing the dataset

plt.scatter(X,Y)
plt.plot(X,regressor.predict(X),color='red')
plt.title('Decision Tree Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()

X.dtype()

#code for visualizing a higher and smoother curve
#this helps to visualize the data being split 
#this section of code wont work if the no of independent variables is more than 2
X_grid = np.arange(min(X) , max(X) , 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y)
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.title('Decision Tree Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


#using the model created to predict the sample value given
X = [[7.5]]
y_pred = regressor.predict(X)
print(y_pred)

#op/: array([200000.])


