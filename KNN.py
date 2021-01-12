# -*- coding: utf-8 -*-
"""


@author: Perault Gael & Bourhim Walid
"""
print(' Let s find the best classification model in order to forecast the missing observations class')

print( 'idea = maximizing accuracy criteria ? ')


print ( ' ----------------------------knn method -----------------------------')


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import math

Data_train = pd.read_csv('Data.csv',sep = ',',header = None)
Data_test = pd.read_csv('TestData.csv',sep = ',', header = None)  # our final test
print(Data_train)
print(Data_train.describe())
X = Data_train.iloc[:,:25]  # Matrice des variables ( Features matrix)
y = Data_train.iloc[:,25].values  # Matrice des classes

#The below script splits the dataset into 80% train data and 20% test data.
from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 40) 

#the following script performs feature scaling: Standardize features by removing the mean and scaling to unit variance
from sklearn.preprocessing import StandardScaler  
scaler = StandardScaler()  
scaler.fit(X_train)

X_train = scaler.transform(X_train)  
X_test = scaler.transform(X_test) 

import seaborn as sns
#option 1: change figure size
plt.figure(figsize=(12,4)) 
# this creates a figure 12 inch wide, 4 inch high
ax = sns.countplot(x =25 ,data=Data_train, palette="Greens_d")
# Set title
ax.set_title("countplot")
plt.show()

print('\n')
print('\n')
print('lets start with K = ', round(math.sqrt(len(X_train))), 'the square root of the Data_train length ')
print('\n')
print('\n')
from sklearn.neighbors import KNeighborsClassifier  
classifier = KNeighborsClassifier(n_neighbors= 63, weights = 'distance')  
classifier.fit(X_train, y_train)  

#The final step is to make predictions on our test data. To do so, execute 
#the following script:

y_pred = classifier.predict(X_test) 
print(y_pred)

#For evaluating an algorithm, confusion matrix, precision, recall and f1 score 
#are the most commonly used metrics. The confusion_matrix and 
#classification_report methods of the sklearn.metrics can be used to calculate 
#these metrics. Take a look at the following script:
from sklearn.metrics import classification_report, confusion_matrix 
print('\n') 
print('--------------confusion_matrix--------------------------')
print('\n')
print(confusion_matrix(y_test, y_pred)) 
print('\n')
print('--------------classification_statistics--------------------------') 
print('\n')
print(classification_report(y_test, y_pred))
print('\n')
print('--------------Accuracy_score--------------------------') 
print('\n')
from sklearn.metrics import accuracy_score
print('For k = 63, the accuracy score is : ', accuracy_score(y_test, y_pred))

#In this section, we will plot the mean error for the predicted values of test 
#set for all the K values between 1 and 70.
#To do so, let's first calculate the mean of error for all the predicted values
# where K ranges from 1 and 80. Execute the following script:
error = []

# Calculating error for K values between 1 and 80
for i in range(1, 80, 2):  
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error.append(np.mean(pred_i != y_test))

#The next step is to plot the error values against K values. 
#Execute the following script to create the plot:
print('\n')
print('\n')
print(' here is show the following error values against k values')
plt.figure(figsize=(12, 6))  
plt.plot(range(1, 80, 2), error, color='red', linestyle='dashed', marker='o',  
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')  
plt.xlabel('K Value')  
plt.ylabel('Mean Error')  
plt.show()

###########################################################
#10-fold cross validation
###########################################################
from sklearn.model_selection import cross_val_score
# creating odd list of K for KNN
k_range = list(range(1, 80, 2))

# empty list that will hold cv scores
cv_scores = []

# perform 10-fold cross validation
for k in k_range:
    print(k)
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=10, scoring='accuracy', n_jobs=-1)
    cv_scores.append(scores.mean())

# changing to misclassification error
MSE = [1 - x for x in cv_scores]
print(MSE)

# determining best k
# in essence, this is basically running the k-fold cross-validation method 80 times
# because we want to run through K values from 1 to 80
# we should have 80 scores here
print('Length of list', len(cv_scores))
print('Max of list', max(cv_scores))

optimal_k = MSE.index(min(MSE)) * 2 + 1
print("The optimal number of neighbors is %d" % optimal_k)

# plot how accuracy changes as we vary k


# plot the value of K for KNN (x-axis) versus the cross-validated accuracy (y-axis)
# plt.plot(x_axis, y_axis)
plt.plot(k_range, cv_scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Cross-validated accuracy')
plt.show()

# plot misclassification error vs k
plt.plot(k_range, MSE)
plt.xlabel('Number of Neighbors K')
plt.ylabel('Misclassification Error')
plt.show()

from sklearn.model_selection import LeaveOneOut
from sklearn import model_selection

knn = KNeighborsClassifier(n_neighbors= optimal_k)
loocv = model_selection.LeaveOneOut() 
results = model_selection.cross_val_score(knn, X, y, cv=loocv, n_jobs=-1) 
len(results)
print('\n')
print('\n')
print('for k = 63, the Accuracy score with the Leave One Out cross validation is better')
print("The Accuracy for the optimal number of neighbors is %.3f%% (%.3f%%)" % (results.mean()*100.0, results.std()*100.0)) 
print("The prediction for the unclassified data:")
knn.fit(X_train, y_train)  
y_pred = knn.predict(Data_test)
print(y_pred)




