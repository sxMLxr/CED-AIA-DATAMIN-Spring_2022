#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 12 14:26:25 2022

@author: skciller
"""

from sklearn import metrics
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt

from sklearn import datasets

# set a seed for reproducibility
random_seed = 25
np.random.seed(random_seed)
# Read the iris dataset and translate to pandas dataframe
bc_sk = datasets.load_breast_cancer()
# Note that the "target" attribute is species, represented as an integer
bc_data = pd.DataFrame(data= np.c_[bc_sk['data'], bc_sk['target']],columns= list(bc_sk['feature_names'])+['target'])

#print(bc_data.head())

from sklearn.model_selection import train_test_split
# The fraction of data that will be test data
test_data_fraction = 0.10

bc_features = bc_data.iloc[:,0:-1]
bc_labels = bc_data["target"]
X_train, X_test, Y_train, Y_test = train_test_split(bc_features, bc_labels, test_size=test_data_fraction,  random_state=random_seed)

# First, let's have a baseline non-boosted decision tree to compare against.
from sklearn.tree import DecisionTreeClassifier
gini_tree = DecisionTreeClassifier(criterion = "gini", random_state=random_seed).fit(X=X_train, y=Y_train)


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
predicted_y = gini_tree.predict(X_test)

print(classification_report(predicted_y,Y_test))

confusion_matrix(predicted_y,Y_test)
'''
array([[17,  3],
       [ 3, 34]])
'''

# Now, let's get boosting. The default estimator for AdaBoost is a *decision stump*. 
# (Remember: a decision stump is simply a decision tree with a height of 1).

from sklearn.ensemble import AdaBoostClassifier

DS = AdaBoostClassifier()

# fit the adaboost classifier
dsfit = DS.fit(X_train, Y_train)

#predict the labels for the test data
pred_y = dsfit.predict(X_test)
#print its performance
print(classification_report(pred_y, Y_test))
#
#              precision    recall  f1-score   support
#
#         0.0       0.95      0.95      0.95        20
#         1.0       0.97      0.97      0.97        37
#
#    accuracy                           0.96        57
#   macro avg       0.96      0.96      0.96        57
#weighted avg       0.96      0.96      0.96        57


#print Confusion matrix
confusion_matrix(pred_y, Y_test)
#array([[19,  1],
#       [ 1, 36]])