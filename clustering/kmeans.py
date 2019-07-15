#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the data
df = pd.read_csv('Mall_Customers.csv')

df.head()

#taking columns 3 and 4 as features
X = df.iloc[:, [3,4]].values

#use the elbow method to find optimal no of clusters
#the for loop finds the inertia (Sum of squared distances of samples to their closest cluster center) and then the value is appended to the list
#then the graph is plot to find the optimal no of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    classifier = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0 )
    classifier.fit(X)
    wcss.append(classifier.inertia_)
plt.plot(1,11, wcss)

op: 4 is the optimal no of clusters

#classifier is fitted on the data
classifier = KMeans(n_clusters = 4, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 0 )
y_pred = classifier.fit_predict(X)

#this plots the data according to the clusters 
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s=50, c = 'red' , label = 'Cluster 1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 50, c = 'blue' , label = 'Cluster 2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 50, c = 'green' , label = 'Cluster 3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s= 50, c = 'yellow' , label = 'Cluster 4')
#this line plots the centroids of the clusters
plt.scatter(classifier.cluster_centers_[:, 0], classifier.cluster_centers_[:, 1], s=300, c= 'magenta', label = 'centroids')
plt.legend()
plt.show()

