import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')
df

X = df.iloc[:, 1:2].values
Y = df.iloc[:, 2].values


#fit the regressor to data
#random_forest_regressor is a collection of decision trees that is used for regression
#changing the no of estimators in the object of the class, we can improve the accuracy of the model
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X,Y)


#code for visualizing a higher and smother curve
X_grid = np.arange(min(X) , max(X) , 0.01)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,Y)
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.title('Decision Tree Regression')
plt.xlabel('Position')
plt.ylabel('Salary')
plt.show()


#predicting the y value of the given sample
X = [[6.5]]
y_pred = regressor.predict(X)
y_pred

o/p: array([160333.33333333])
