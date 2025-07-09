#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 19 11:25:07 2022

@author: skciller
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets


#========= model information
from pomegranate import *
#First, we define our top level nodes with their base probabilities.

visit_to_asia = DiscreteDistribution({'T':0.01, 'F':0.99})
smoke = DiscreteDistribution({'T':0.5, 'F':0.5})

# Now, we have to fill in all of the conditional probability tables for the other nodes

has_tb = ConditionalProbabilityTable(
    [
        #Asia? #HasTB #Probability
        ["T","T",0.05],
        ["T","F",0.95],
        
        ["F","T",0.01],
        ["F","F",0.99],
    ], [visit_to_asia])


has_lung_cancer = ConditionalProbabilityTable(
    [
        #Smoke? 
        ["T","T",0.1],
        ["T","F",0.9],
        
        ["F","T",0.01],
        ["F","F",0.99]
    ], [smoke])


has_bc = ConditionalProbabilityTable(
    [
        #Smoke?
        ["T","T",0.6],
        ["T","F",0.4],
        
        ["F","T",0.3],
        ["F","F",0.7]
    ], [smoke])

tb_or_cancer = ConditionalProbabilityTable(
    [
        #Lung? TB? 
        ["T","T","T",1],
        ["T","T","F",0],
        
        ["T","F","T",1],
        ["T","F","F",0],
        
        ["F","T","T",1],
        ["F","T","F",0],
        
        ["F","F","T",0],
        ["F","F","F",1]
    ], [has_lung_cancer,has_tb])

x_ray_abnormal = ConditionalProbabilityTable(
    [
        #TB or Cancer?
        ["T","T",0.98],
        ["T","F",0.02],
        
        ["F","T",0.05],
        ["F","F",0.95]
    ], [tb_or_cancer])

dyspnea = ConditionalProbabilityTable(
    [
        #BC
        ["T","T","T",0.9],
        ["T","T","F",0.1],
        
        ["T","F","T",0.8],
        ["T","F","F",0.2],
        
        ["F","T","T",0.7],
        ["F","T","F",0.3],
        
        ["F","F","T",0.1],
        ["F","F","F",0.9]
    ], [has_bc, tb_or_cancer])

# Next we have to create all the nodes
asia_node = Node(visit_to_asia, name="asia")
tb_node = Node(has_tb, name="tb")
smoke_node = Node(smoke, name="smoke")
lung_node = Node(has_lung_cancer, name="lung")
bronc_node = Node(has_bc, name="bc")
either_node = Node(tb_or_cancer, name="either")
xray_node = Node(x_ray_abnormal,name="xray")
dysp_node = Node(dyspnea, name="dysp")

# Now we init the model
model = BayesianNetwork("ASIA")
model.add_states(asia_node,
                 tb_node,
                 smoke_node,
                 lung_node,
                 bronc_node,
                 either_node,
                 xray_node,
                 dysp_node)

# Add all of the correct edges 
model.add_edge(asia_node, tb_node)

model.add_edge(smoke_node, bronc_node)
model.add_edge(smoke_node, lung_node)

model.add_edge(tb_node,either_node)
model.add_edge(lung_node,either_node)

model.add_edge(either_node, xray_node)
model.add_edge(either_node, dysp_node)

model.add_edge(bronc_node, dysp_node)

# And then commit our changes
model.bake()

# Helper function to print the model structure
def print_model_structure(model, features):
    for i in range(len(features)):
        parents = [features[pi] for pi in model.structure[i]]
        print(f'Node "{features[i]}" has parents: {parents}')

# We'll keep our features in this order for consistency
features = [
    "Visit to Asia",
    "Has TB",
    "Smoker",
    "Has Lung Cancer",
    "Has Bronchitis",
    "TB or Cancer",
    "XRay Abnormal",
    "Dyspnea"
]
#========= model information 
##Evaluating Bayes Net 
#Let's see how well this net is on inferencing from data. We're going to remove the Bronchitis column from this dataset, and see if our net can predict what the missing value should be.

# Some data we will use to generate our probabilities

asia_data = pd.read_csv("ASIA10k.csv")
#cli verification of size:  asia_data.shape
'''
print(asia_data.head())
Out[4]: 
  asia tub smoke lung bronc either xray dysp
0   no  no    no   no    no     no   no  yes
1   no  no   yes   no    no     no   no   no
2   no  no    no   no    no     no   no   no
3   no  no   yes   no    no     no   no   no
4   no  no   yes   no   yes     no   no  yes
'''

#lets make sure were consistant with our labels
asia_data = asia_data.replace("no", "F").replace("yes", "T")
'''
Out[6]: 
  asia tub smoke lung bronc either xray dysp
0    F   F     F    F     F      F    F    T
1    F   F     T    F     F      F    F    F
2    F   F     F    F     F      F    F    F
3    F   F     T    F     F      F    F    F
4    F   F     T    F     T      F    F    T
'''

values = asia_data.values.copy()
indices = np.random.choice(asia_data.index, 1000)
values = values[indices]
#print(values[:10])
#change column 4 to NONE so we can predict this variables outcome.
values[:,4] = None
#confirm out changes:  print(values[1])
'''Out[8]: array(['F', 'F', 'F', 'F', None, 'F', 'F', 'T'], dtype=object)'
'''

preds = model.predict(values)   #define our model, predict using "test data"

#build confusion matrix and test for accuracy of
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

#first confusion matrix
print(confusion_matrix(asia_data.values[indices,4], np.array(preds)[:,4]))

print(classification_report(asia_data.values[indices,4], np.array(preds)[:,4]))