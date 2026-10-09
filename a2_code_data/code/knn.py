"""
Implementation of k-nearest neighbours classifier
"""

import numpy as np

import utils
from utils import euclidean_dist_squared
from scipy import stats 

class KNN:
    X = None
    y = None

    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X = X  # just memorize the training data
        self.y = y

    def predict(self, X_hat):
        dist = euclidean_dist_squared(X_hat, self.X)
    
        nearest_indices = np.argsort(abs(dist))[:, :self.k]
    
        nearest_neighbor = self.y[nearest_indices]
    
        common_label = stats.mode(nearest_neighbor, axis=1) # from scipy import stats 
        predictions = common_label.mode.squeeze()
        
        return np.array(predictions)
