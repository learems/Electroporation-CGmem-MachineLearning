# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 12:15:51 2021

@author: Xinru & Fangwei
"""

import numpy as np
import os
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.neural_network import MLPClassifier
from sklearn import svm

class MLAnalysis:
    
     def  __init__(self,X_train,Y_train,X_test,Y_test):
        
        self.X_train = X_train
        
        self.Y_train = Y_train
        
        self.X_test = X_test
        
        self.Y_test = Y_test
        
        self.train_accuracy = 0.0
        
        self.test_accuracy = 0.0
        
        self.prediction_test = -np.ones(len(self.Y_test))
        
        
     @classmethod
     def PrepareData(cls,X_APM,Y_APM,X_BPM,Y_BPM,pattern='mix'):
         
         if pattern == 'aa':
            X_smote, Y_smote = MLAnalysis.Oversampling(X_APM,Y_APM)
            X_train, X_test, Y_train, Y_test = MLAnalysis.Split_train_test(X_smote,Y_smote)
            
         if pattern == 'bb':
            X_smote, Y_smote = MLAnalysis.Oversampling(X_BPM,Y_BPM)
            X_train, X_test, Y_train, Y_test = MLAnalysis.Split_train_test(X_smote,Y_smote)
        
         if pattern == 'ab':
            X_train, Y_train = MLAnalysis.Oversampling(X_APM,Y_APM)
            X_test = X_BPM
            Y_test = Y_BPM
            
         if pattern == 'mix':
            X_smote_APM, Y_smote_APM = MLAnalysis.Oversampling(X_APM,Y_APM)
            X_smote_BPM, Y_smote_BPM = MLAnalysis.Oversampling(X_BPM,Y_BPM)
            X_mix = pd.concat([X_smote_APM,X_smote_BPM],axis=0) 
            Y_mix = np.concatenate([Y_smote_APM,Y_smote_BPM],axis=0)
            X_train, X_test, Y_train, Y_test = MLAnalysis.Split_train_test(X_mix,Y_mix)
         
         return cls(X_train, Y_train, X_test, Y_test)

     @classmethod  
     def Oversampling(cls,X,Y):
        
        smote = SMOTE()
        
        X_smote, Y_smote = smote.fit_sample(X.astype('float'),Y)

        return X_smote, Y_smote
    
     '''
     def Shuffle(self):
             
         pass
     '''
     
     @classmethod
     def Split_train_test(cls,X,Y):
        
        X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.4,random_state=10)

        return X_train, X_test, Y_train, Y_test
    
    
     def ML(self,model='RF'):
        if model == 'RF':
            self.model = RandomForestClassifier(random_state=20)
        if model == 'SVM':
            self.model = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
        if model == 'NN':
            self.model = MLPClassifier(hidden_layer_sizes=(len(self.X_train.columns),len(self.X_train.columns),len(self.X_train.columns)),max_iter=2000)
        
        self.model.fit(self.X_train,self.Y_train)

        self.prediction_test = self.model.predict(self.X_test)

        self.train_test = self.model.predict(self.X_train)
        
        self.train_accuracy = metrics.accuracy_score(self.Y_train,self.train_test)
        
        self.test_accuracy = metrics.accuracy_score(self.Y_test,self.prediction_test)

        print('test_Accuracy = ', self.test_accuracy)

        print('train_Accuracy = ', self.train_accuracy)
        
        return self
        



