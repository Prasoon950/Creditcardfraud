# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:20:16 2020

@author: DELL
"""

import numpy as np 
import pandas as pd 


ex = pd.read_csv("F:\zipped\creditcard.csv")
exe = ex.sample(frac = 0.1,random_state=1)

fraud = exe[exe['Class']==1]
normal = exe[exe['Class']==0]

outliers = len(fraud)/float(len(normal))


state =np.random.RandomState(42)

X = exe.iloc[:,:]
Y = exe.iloc[:,-1:]

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)


from sklearn.metrics import accuracy_score
from sklearn.ensemble import IsolationForest

clf = IsolationForest(max_samples=100, random_state=state)
clf.fit(X_train)
   

 
y_pred = clf.predict(X_train)
y_pred = clf.predict(X_test)
outliers = state.uniform(low=-1, high=5, size=(50, 31))
y_pred_outliers = clf.predict(outliers)

y_pred[y_pred == 1] = 0
y_pred[y_pred == -1] = 1




from sklearn.metrics import confusion_matrix

accuracy=confusion_matrix(Y_test,y_pred)
accuracy

from sklearn.metrics import accuracy_score

accuracy=accuracy_score(Y_test,y_pred)
accuracy



   
        
    


        
        
        
