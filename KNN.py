# Simple implementation of KNN using the Euclidean Distance 
import numpy as np
from collections import Counter

def euclidean_distnace(x, y):
    return np.sqrt(np.sum((x-y)**2)) 


class KNN(): 
    def __init__(self, k=3): # default value for k 
        self.k = k

    def fit(self, X, y): 
        self.X_train = X
        self.y_train = y

    def predict(self, X): 
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x): 
        # compute euclidean distance 
        distances = [euclidean_distnace(x, x_train) for x_train in self.X_train]

        # get the indices of the k nearest neighbors 
        k_nearest_indices = np.argsort(distances)[:self.k] # getting the first k
        k_nearest_lables = [self.y_train[i] for i in k_nearest_indices]

        # get the vote/prediction

        most_common_label = Counter(k_nearest_lables).most_common()
        return most_common_label[0][0]



