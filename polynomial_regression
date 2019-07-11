import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('Position_Salaries.csv')
df.head()

X = df.iloc[:,1:2].values
y = df.iloc[:,-1].values

#fitting linear regression model
from sklearn.linear_model import LinearRegression
linear_reg = LinearRegression()
linear_reg.fit(X,y)


#fitting polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
X_poly

lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly,y)

#visualing the linear model
plt.scatter(X,y,color = 'red')
plt.plot(X,linear_reg.predict(X), color = 'blue')
plt.title('LinearModel')

#visualing the poly model
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X,lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
plt.title('PolyModel')
