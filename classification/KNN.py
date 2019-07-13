#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing dataframe
df = pd.read_csv('Social_Network_Ads.csv')


#taking column 2 and 3 as features and the last column as the label column
X = df.iloc[:, 2:4].values
y = df.iloc[:,-1].values

#feature scaling 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

X = scaler.fit_transform(X)

#splitting the data into training and test set
from sklearn.model_selection import train_test_split

xtrain,xtest,ytrain,ytest = train_test_split(X, y , test_size = 0.25, random_state = 0)

#importing the classifier and creating an object of the classifier
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)

#fitting the classifier on the training data
classifier.fit(xtrain,ytrain)

#predicting using the classifier
y_pred = classifier.predict(xtest)

#checking the result by comparing the labels with the predicted values
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(ytest, y_pred)

#visualizing the training data
from matplotlib.colors import ListedColormap

X_set, yset = xtrain, ytrain

X1,X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))

plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i, j in enumerate(np.unique(yset)):
    plt.scatter(X_set[yset == j, 0], X_set[yset == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()


#calculating the classifier accuracy
classifier.score(xtrain,ytrain)
o/p:  0.9166666666666666

#calculating the classifier score on the testing data-set
classifier.score(xtest,ytest)
o/p: 0.93
