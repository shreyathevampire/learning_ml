from sklearn import tree
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn import svm
from sklearn import neighbors
#[height,weight,shoe-size]

X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40],
     [190, 90, 37], [175, 64, 39],
[177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female',
'female', 'male', 'male']

clf = tree.DecisionTreeClassifier()

clf1 = linear_model.LogisticRegression()

clf2 = linear_model.SGDClassifier(max_iter=10)

clf3 = svm.SVC(gamma='auto')

clf4 = neighbors.KNeighborsClassifier(n_neighbors=3)

clf = clf.fit(X,Y)

clf1 = clf1.fit(X,Y)

clf2 = clf2.fit(X,Y)

clf3 = clf3.fit(X,Y)

clf4 = clf4.fit(X,Y)

prediction = clf.predict([[190,70,43],[160,70,37]])

prediction1 = clf1.predict([[190,70,43],[160,70,37]])

prediction2 = clf2.predict([[190,70,43],[160,70,37]])

prediction3 = clf3.predict([[190,70,43],[160,70,37]])

prediction4 = clf4.predict([[190,70,43],[160,70,37]])

print ( "Decision - tree", prediction)
print ("Logistic-Regression", prediction1)
print ( "Stochastic Gradient-Descent", prediction2)
print ( "SVM", prediction3)
print ( "K-nearest-neighbors", prediction4)
#print (accuracy_score(clf,prediction))
