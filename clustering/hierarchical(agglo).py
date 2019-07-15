#importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#reading the data into a dataframe
df = pd.read_csv('Mall_Customers.csv')

df.head()

X = df.iloc[:, [3,4]].values



# find the optimal no of clusters that can be formed 
#dendogram is used for this purpose
from scipy.cluster.hierarchy import dendrogram, linkage

dn = dendrogram(linkage(X, 'ward'))

#op: 5 is the optimal no of clusters 
#importing clustering algorithm and fitting it on the data
from sklearn.cluster import AgglomerativeClustering

clustering = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_pred = clustering.fit_predict(X)

#visualizing the data on a 2D plot
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s= 30, c='red', label = 'Cluster-1')
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s= 30, c='blue', label = 'Cluster-2')
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s= 30, c='black', label = 'Cluster-3')
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s= 30, c='orange', label = 'Cluster-4')
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s= 30, c='green', label = 'Cluster-5')
plt.xlabel('Income')
plt.ylabel('Spending')

