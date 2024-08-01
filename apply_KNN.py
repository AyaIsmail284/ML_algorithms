# Applying KNN to predict a value based on the IRIS data
from KNN import KNN
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split

cmap = ListedColormap(['#FF0000','#00FF00','#0000FF'])

iris = datasets.load_iris()
X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# plt.figure()
# plt.scatter(X[:,2],X[:,3], c=y, cmap=cmap, edgecolor='k', s=20)
# plt.show()

KNN_clf = KNN(k=7)
KNN_clf.fit(X_train, y_train)

predictions = KNN_clf.predict(X_test)

print(predictions)

# estimating the accuracy of the prediction 
accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)