from random_stump import RandomStumpInfoGain
from decision_tree import DecisionTree
import numpy as np
from scipy import stats 

import utils


class RandomTree(DecisionTree):
    def __init__(self, max_depth):
        DecisionTree.__init__(
            self, max_depth=max_depth, stump_class=RandomStumpInfoGain
        )

    def fit(self, X, y):
        n = X.shape[0]
        boostrap_inds = np.random.choice(n, n, replace=True)
        bootstrap_X = X[boostrap_inds]
        bootstrap_y = y[boostrap_inds]

        DecisionTree.fit(self, bootstrap_X, bootstrap_y)


class RandomForest:
    """
    YOUR CODE HERE FOR Q4
    Hint: start with the constructor __init__(), which takes the hyperparameters.
    Hint: you can instantiate objects inside fit().
    Make sure predict() is able to handle multiple examples.
    """
    
    def __init__(self, num_trees, max_depth):
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.trees = []
    
    def fit(self, X, y):
        for tree in range(self.num_trees):
            tree = RandomTree(max_depth=self.max_depth)
    
            tree.fit(X, y)
            self.trees.append(tree)
    
    
    def predict(self, X_pred):
        predictions = np.zeros((len(X_pred), self.num_trees))
        
        for i, tree in enumerate(self.trees):
            predictions[:, i] = tree.predict(X_pred)
        
        return stats.mode(predictions, axis=1).mode.squeeze()