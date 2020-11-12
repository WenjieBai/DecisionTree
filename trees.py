#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from collections import Counter
import math
import sys


# In[2]:


class Node:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


# In[3]:


class DecisionTreeClassifier:
    def __init__(self, max_depth=8, max_example = 50):
        self.max_depth = max_depth
        self.max_example = max_example
        
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.sum(y == y_hat)/y.size

    def _gini(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        for idx in range(self.n_features_):
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr
    

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )


        if (depth <= self.max_depth) and (y.size >= self.max_example):
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node
    
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


# In[4]:


class BaggingDecisionTree:
    def __init__(self, n_trees=30, max_depth_outer=8, max_example = 50):
        self.n_trees_ = n_trees
        self.max_depth_outer = max_depth_outer
        self.max_example = max_example
        
    def fit(self, X, y):
        models = []
        data = np.c_[X,y]
        for i in range(self.n_trees_):
            data_sample = data[np.random.choice(X.shape[0], X.shape[0], replace=True), :]
            X_s = data_sample[:, :-1]
            y_s = data_sample[:, -1]
            clf = DecisionTreeClassifier(max_depth = self.max_depth_outer)
            clf.fit(X_s, y_s)
            models.append(clf)
            print(f'model {i} trained')
        self.models = models
        
    def predict(self, X):
        y_all_models = np.empty([X.shape[0], self.n_trees_])
        for i in range(self.n_trees_):
            y_all_models[:,i] = self.models[i].predict(X)
            
        y_hat = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            y_hat[i] = self.find_majority(y_all_models[i,:])
        return y_hat

    def find_majority(self, votes):
        vote_count = Counter(votes)
        top_two = vote_count.most_common(2)
        if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
            return 0
        return top_two[0][0]
        
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.sum(y == y_hat)/y.size


# In[11]:


class RandomForest:
    def __init__(self, n_trees=30):
        self.n_trees_ = n_trees
        
    def fit(self, X, y):
        models = []
        data = np.c_[X,y]
        for i in range(self.n_trees_):
            data_sample = data[np.random.choice(X.shape[0], X.shape[0], replace=True), :]
            X_s = data_sample[:, :-1]
            y_s = data_sample[:, -1]
            clf = DecisionTreeSampleFeature()
            clf.fit(X_s, y_s)
            models.append(clf)
#             print(f'model {i} trained')
        self.models = models
        
    def predict(self, X):
        y_all_models = np.empty([X.shape[0], self.n_trees_])
        for i in range(self.n_trees_):
            y_all_models[:,i] = self.models[i].predict(X)
            
        y_hat = np.empty(X.shape[0])
        for i in range(X.shape[0]):
            y_hat[i] = self.find_majority(y_all_models[i,:])
        return y_hat
        

    def find_majority(self, votes):
        vote_count = Counter(votes)
        top_two = vote_count.most_common(2)
        if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
            return 0
        return top_two[0][0]
        
    def score(self, X, y):
        y_hat = self.predict(X)
        return np.sum(y == y_hat)/y.size


# In[6]:


class DecisionTreeSampleFeature:
    def __init__(self, max_depth=8, max_example = 50):
        self.max_depth = max_depth
        self.max_example = max_example
        
    def fit(self, X, y):
        self.n_classes_ = len(set(y))
        self.n_features_ = X.shape[1]
        self.tree_ = self._grow_tree(X, y)

    def predict(self, X):
        return [self._predict(inputs) for inputs in X]

    def score(self, X, y):
        y_hat = self.predict(X)
        return np.sum(y == y_hat)/y.size

    def _gini(self, y):
        m = y.size
        return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in range(self.n_classes_))

    def _best_split(self, X, y):
        m = y.size
        if m <= 1:
            return None, None

        num_parent = [np.sum(y == c) for c in range(self.n_classes_)]

        best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
        best_idx, best_thr = None, None

        sample_features = np.random.choice(range(self.n_features_),math.isqrt(self.n_features_), replace=False)
        
        for idx in sample_features:
            thresholds, classes = zip(*sorted(zip(X[:, idx], y)))

            num_left = [0] * self.n_classes_
            num_right = num_parent.copy()
            for i in range(1, m):  # possible split positions
                c = classes[i - 1]
                num_left[c] += 1
                num_right[c] -= 1
                gini_left = 1.0 - sum(
                    (num_left[x] / i) ** 2 for x in range(self.n_classes_)
                )
                gini_right = 1.0 - sum(
                    (num_right[x] / (m - i)) ** 2 for x in range(self.n_classes_)
                )

                gini = (i * gini_left + (m - i) * gini_right) / m

                if thresholds[i] == thresholds[i - 1]:
                    continue

                if gini < best_gini:
                    best_gini = gini
                    best_idx = idx
                    best_thr = (thresholds[i] + thresholds[i - 1]) / 2
        return best_idx, best_thr
    

    def _grow_tree(self, X, y, depth=0):
        num_samples_per_class = [np.sum(y == i) for i in range(self.n_classes_)]
        predicted_class = np.argmax(num_samples_per_class)
        node = Node(
            gini=self._gini(y),
            num_samples=y.size,
            num_samples_per_class=num_samples_per_class,
            predicted_class=predicted_class,
        )


        if (depth <= self.max_depth) and (y.size >= self.max_example):
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = idx
                node.threshold = thr
                node.left = self._grow_tree(X_left, y_left, depth + 1)
                node.right = self._grow_tree(X_right, y_right, depth + 1)
        return node
    
    def _predict(self, inputs):
        node = self.tree_
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        return node.predicted_class


# In[7]:


def decisionTree(trainingSet, testSet):
    X_train = trainingSet[:, :-1]
    y_train = trainingSet[:, -1]
    X_test = testSet[:,:-1]
    y_test = testSet[:, -1]
    clf = DecisionTreeClassifier()
    clf.fit(X_train,y_train)
    print(f'Training Accuracy DT: {clf.score(X_train,y_train)}')
    print(f'Testing Accuracy DT: {clf.score(X_test,y_test)}')
    


# In[8]:


def bagging(trainingSet, testSet):
    X_train = trainingSet[:, :-1]
    y_train = trainingSet[:, -1]
    X_test = testSet[:,:-1]
    y_test = testSet[:, -1]
    clf = BaggingDecisionTree()
    clf.fit(X_train,y_train)
    print(f'Training Accuracy BT: {clf.score(X_train,y_train)}')
    print(f'Testing Accuracy BT: {clf.score(X_test,y_test)}')


# In[9]:


def randomForests(trainingSet, testSet):
    X_train = trainingSet[:, :-1]
    y_train = trainingSet[:, -1]
    X_test = testSet[:,:-1]
    y_test = testSet[:, -1]
    clf = RandomForest()
    clf.fit(X_train,y_train)
    print(f'Training Accuracy RF: {clf.score(X_train,y_train)}')
    print(f'Testing Accuracy RF: {clf.score(X_test,y_test)}')


# In[12]:


def main():
    if len(sys.argv) != 4:
    	raise Exception('except 4 arguments: trees.py trainingSet.csv testSet.csv 1(2 or 3)')

    print(len(sys.argv))
    trainSet = pd.read_csv('trainingSet.csv')
    testSet = pd.read_csv('testSet.csv')
    df_train = trainSet.to_numpy()
    df_test = testSet.to_numpy()

    mode = int(sys.argv[3])
    df_train = pd.read_csv('trainingSet.csv').to_numpy()
    df_test = pd.read_csv('testSet.csv').to_numpy()
    if mode == 1:
        decisionTree(df_train,df_test)
    elif mode == 2:
        bagging(df_train,df_test)
    elif mode == 3:
        randomForests(df_train,df_test)
    else:
        print('the last argument can either be 1, 2, or 3')

if __name__=="__main__": 
    main() 




