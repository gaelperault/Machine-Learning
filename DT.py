# -*- coding: utf-8 -*-
"""


@author: Perault Gael & Bourhim Walid
"""
import os
import numpy as np  
import matplotlib.pyplot as plt  
import pandas as pd 
from sklearn.metrics import confusion_matrix 
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report 
import math

Data_train = pd.read_csv('Data.csv',sep = ',',header = None)
Data_test = pd.read_csv('TestData.csv',sep = ',', header = None)  # our final test
 # Matrice des classes

###############################################################################
###############################################################################
################ decision trees   ########################
###############################################################################

# Function to split the dataset 
def splitdataset(input_data): 
  
    # Seperating the target variable 
    X = Data_train.iloc[:,:25]
    y = Data_train.iloc[:,25].values
  
    # Spliting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split(  
    X, y, test_size = 0.2) 
      
    return X, y, X_train, X_test, y_train, y_test 
      

# Function to perform training with giniIndex. 
def train_using_gini_depth2(X_train,y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=2, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
      
  
# Function to make predictions 
def prediction(X_test, clf_object): 
  
    # Predicton on test with giniIndex 
    y_pred = clf_object.predict(X_test) 
    print("Predicted values:") 
    print(y_pred) 
    return y_pred 
      
# Function to calculate accuracy 
def cal_accuracy(y_test, y_pred): 
      
    print("Confusion Matrix: ")
    print('\n')
    print(confusion_matrix(y_test, y_pred) )
    print('\n')
      
    print ("Accuracy : ")
    print('\n')
    print(accuracy_score(y_test,y_pred)*100)
    print('\n')
      
    print("Report : ")
    print('\n')
    print(classification_report(y_test, y_pred))
    print('\n')
    
#################################################################################
#################################################################################
  
X, y, X_train, X_test, y_train, y_test = splitdataset(Data_train) 
clf_gini = train_using_gini_depth2(X_train, y_train) 


print('------------------statistics ')
# Operational Phase 
print("----------------------------Prediction Using Gini Index:----------------------------") 
print('\n')
      
# Prediction using gini 
y_pred_gini = prediction(X_test, clf_gini) 
print('----------------------------Accuracy using Gini Index----------------------------')
cal_accuracy(y_test, y_pred_gini) 
print('\n')
      

###visualize tree
import graphviz
from sklearn import tree
tree.export_graphviz(clf_gini,out_file='tree.dot') 

import pydot
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')


from IPython.display import Image
Image(filename = 'tree.png')

print('\n')
print('\n')
print('\n')
print('-----------------------lets implement our results using a larger depth > 2 -----------------------')
print('\n')
print('------------------statistics using a depth = 3 gini index -----------------')
print('\n')    
###############################################################################
print('\n')
print('\n')
def train_using_gini_depth3(X_train,y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=3, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 

clf_gini = train_using_gini_depth3(X_train, y_train) 
print('\n')
y_pred_gini = prediction(X_test, clf_gini) 
print('\n')
cal_accuracy(y_test, y_pred_gini) 
print('\n')

###visualize tree
import graphviz
from sklearn import tree
tree.export_graphviz(clf_gini,out_file='tree_depth3.dot') 

import pydot
(graph,) = pydot.graph_from_dot_file('tree_depth3.dot')
graph.write_png('tree_depth3.png')

from IPython.display import Image
Image(filename = 'tree_depth3.png')

print('\n')
print('------------------statistics using a depth = 4 gini index -----------------')

def train_using_gini_depth4(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=4, min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
clf_gini = train_using_gini_depth4(X_train, X_test, y_train) 
print('\n')
y_pred_gini = prediction(X_test, clf_gini) 
print('\n')
cal_accuracy(y_test, y_pred_gini) 
print('\n')
print('\n')
###visualize tree


import graphviz
from sklearn import tree
tree.export_graphviz(clf_gini,out_file='tree_depth4.dot') 

import pydot
(graph,) = pydot.graph_from_dot_file('tree_depth4.dot')
graph.write_png('tree_depth4.png')

from IPython.display import Image
Image(filename = 'tree_depth4.png')


print('------------------statistics none max_depth gini index-----------------')

def train_using_gini_no_max(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini")
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 
clf_gini = train_using_gini_no_max(X_train, X_test, y_train) 
y_pred_gini = prediction(X_test, clf_gini) 
cal_accuracy(y_test, y_pred_gini)


###visualize tree
import graphviz
from sklearn import tree
tree.export_graphviz(clf_gini,out_file='tree_nodepth.dot') 

import pydot
(graph,) = pydot.graph_from_dot_file('tree_nodepth.dot')
graph.write_png('tree_nodepth.png')

from IPython.display import Image
Image(filename = 'tree_nodepth.png')

print('\n')
print('---------------------------10_k-fold_Cross_validation----------------------')
print('\n')


import pandas as pd
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt 

#k-Fold Cross-Validation
#setting n_jobs=-1 select all physical cores and maximises their usage.
X = Data_train.iloc[:,:25]
X_n = X.apply(lambda x:(x-x.min()) / (x.max()-x.min()))

y = Data_train.iloc[:,25].values
depth_range = range(2,30)
depth = []
for i in depth_range:
    clf = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=i, min_samples_leaf=5)
    # Perform 10-fold cross validation 
    scores = cross_val_score(estimator=clf, X=X_n, y=y, cv=10, n_jobs=1)
    depth.append((scores.mean()))
print(depth)

print('Length of list', len(depth))
print('Max Gini index', max(depth))
# plot how accuracy changes as we vary depth
# plot the value of depth for KNN (x-axis) versus the cross-validated accuracy (y-axis)
plt.plot(depth_range, depth)
plt.xlabel('Value of depth for decision tree')
plt.ylabel('Cross-validated accuracy')
plt.show()
#We can extract the position of max with:
max_gini_position= max(enumerate(depth), key=(lambda x: x[1]))

print('using cross_validation on our decision tree the gini index maximum is reached for a depth equal to', max_gini_position[0])
print('\n')
print( ' We can now process and get the Accuracy for this maximum')

print('----------------------statistics for maximum gini index------------------- ')
print('\n')
print('\n')

def train_using_gini_max(X_train, X_test, y_train): 
  
    # Creating the classifier object 
    clf_gini = DecisionTreeClassifier(criterion = "gini", 
            random_state = 100,max_depth=max_gini_position[0], min_samples_leaf=5) 
  
    # Performing training 
    clf_gini.fit(X_train, y_train) 
    return clf_gini 

print('\n')

BEST_clf_gini = train_using_gini_max(X_train, X_test, y_train) 
print('\n')
y_pred_gini = prediction(X_test, BEST_clf_gini) 
print('\n')
cal_accuracy(y_test, y_pred_gini) 
print('\n')
print('\n')

print('Accuracy for a 10_cross_validation correspond to ',accuracy_score(y_test, y_pred_gini))
print((accuracy_score(y_test, y_pred_gini)) * 100, '% of our data are well classified')

print('\n')
print('\n')
print(' ----------------------Final Prediction for Data_test ----------------------------')
print('\n')
print('\n')

print("The prediction for the unclassified data:")
y_pred = prediction(Data_test, BEST_clf_gini)










