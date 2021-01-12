# -*- coding: utf-8 -*-


"""
Created on Wed Oct  9 16:41:52 2019
@author: Gael Perault & Walid Bourhim
"""


import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import os

os.getcwd()
Data_train = pd.read_csv('Data.csv',sep = ',',header = None)
Data_test = pd.read_csv('TestData.csv',sep = ',', header = None)  # our final test
X = Data_train.iloc[:,:25]
y = Data_train.iloc[:,25].values

# split X and y into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=40)
#normalize 
X_train_n = X_train.apply(lambda x:(x-x.min()) / (x.max()-x.min()))
X_test_n = X_test.apply(lambda x:(x-x.min()) / (x.max()-x.min()))

from sklearn.linear_model import LogisticRegression
# here we have a multi_class so we can use newton-cg solver,‘lbfgs’, ‘sag’, ‘saga’ .
logreg = LogisticRegression(C=1e9,solver='newton-cg',max_iter=100000)

# fit the model with data
logreg.fit(X_train_n,y_train) 
logreg.coef_ 
logreg.intercept_

prediction=logreg.predict(X_test_n) # à comparer avec y_test

# import the metrics class
from sklearn import metrics
cnf_matrix = metrics.confusion_matrix(y_test, prediction)
cnf_matrix
#Let's evaluate the model using model evaluation metrics such as accuracy, precision, and recall.
print("The Accuracy score equal:",metrics.accuracy_score(y_test, prediction))

print("The prediction for the unclassified data:")
y_pred = logreg.predict(Data_test)
print(y_pred)


###############################################################################
######### using cv logistic regression
###############################################################################

from sklearn.linear_model import LogisticRegression
from sklearn import metrics
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegressionCV



clf = LogisticRegressionCV(cv=5, random_state=0, multi_class='ovr').fit(X_train_n,y_train)
clf.score(X_train_n, y_train) 
print(dir(clf))
clf.scores_


###############################################################################
######### using glm linear models
###############################################################################

X_train_n = sm.add_constant(X_train_n)
model = sm.GLM(y_train,X_train_n, family=sm.families.Binomial())
result = model.fit()
print(dir(result))
print(result.summary())

X_test_n = sm.add_constant(X_test_n)
y_pred_proba = result.predict(X_test_n)
y_pred_GLM = y_pred_proba.round()

print("Accuracy:",metrics.accuracy_score(y_test, y_pred_GLM))
print("Precision:",metrics.precision_score(y_test, y_pred_GLM))

fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()
