# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 21:26:03 2019

@author: Lenovo
"""

import pandas as pd
import numpy as np 
from sklearn.preprocessing import LabelEncoder


df=pd.read_csv('features_phase6a.csv')

df.describe()

df.isnull().sum()

'''Analysing the target variable'''

df['LABEL'].describe()
df['LABEL'].value_counts()

'''Changing to numerical categories for further analysing and plotting'''

encoder=LabelEncoder()
encoder.fit(df['LABEL'])
df['TARGET'] = encoder.transform(df['LABEL'])
df['TARGET']

#also generating separate columns for each of the 17 classes to further analyze the covariance of features

# Target Classes
c=['Others', 'abstract', 'authors', 'affiliation',
       'correspondingauthor', 'dummy', 'keywordsdefault', 'articletitle',
       'abstracttitle', 'acknowledgementstitle', 'acknowledgements',
       'grant-sponsor', 'articlefootnote', 'grant-highlight',
       'nomenclature', 'listdefinition', 'grant-number']

df['LABEL'].unique()
dftargetclasses = pd.get_dummies(df['LABEL'], drop_first=False, columns = c)




'''FEATURE SELECTION'''
'''Correlation analysis b/w o/p and featutes'''



dfcombined = pd.concat([df, dftargetclasses], axis=1)

corr=dfcombined.corr()

# Checking the columns where the correlation coef is more than .6 :
co = []
for i in corr.columns:
    for j in corr.index:
        if (corr.get_value(i,j)) > abs(0.5) and i!=j and i in c :
            print(i,j)
            print(corr.get_value(i,j))
            
### From above analysis we see that there is good amoutnt of correlation between the following features and target classes:
'''abstract LEN
0.5609432782213328
abstract DP_DOBJ
0.5061626548661987
abstract DP_PREP
0.5641740010602055
abstract DP_POBJ
0.567864452415253
abstract DP_NSUBJ
0.5290438412596312
abstract CG10
0.5634817847876632
abstract C_TITLE
0.622884270256897
abstract NER_OTH
0.5525204761353709
correspondingauthor C_ABS
0.5338132996169441
keywordsdefault C_KEYW
0.5239717329365035'''

### Since the correlation b/w the features and target classes is not too prominent, a logistic regression classifier might not be a good fit for the task.
## We will also apply chi square, LDA and ANOVA for different set of features later on.

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2, f_classif

x=df.drop(['TARGET', 'LABEL'], axis=1)
y=df['TARGET']

bestfeatures = SelectKBest(score_func=f_classif, k=10)
fit = bestfeatures.fit(x,y)

for i in zip(x.columns, fit.scores_):
    if i[1]>500:
        print(i)








'''Feature importance analysis using a tree based classifier'''

## preparing the data for fitting the model

x=df.drop(['TARGET', 'LABEL'], axis=1)
y=df['TARGET']

## Separating into train and validation set as feature scaling is not required for tree based classifiers

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

## Fitting the model
from sklearn.ensemble import RandomForestClassifier

rf=RandomForestClassifier(n_estimators=100)

rf.fit(x_train,y_train)

y_pred=rf.predict(x_test)


### checking the accuracy score and confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix

print(accuracy_score(y_pred,y_test))
print(confusion_matrix(y_pred,y_test))

## since the rf classifier is giving an accuracy score of .85, checking for the features significant in the classification
for feature in zip(x.columns, rf.feature_importances_):

    if feature[1]>.010:
        print(feature)
            

### sklearn select from model:
from sklearn.feature_selection import SelectFromModel

sfm = SelectFromModel(rf, threshold=0.01)
sfm.fit(x_train, y_train)

for feature_list_index in sfm.get_support(indices=True):
    print(x.columns[feature_list_index])
    
    
    

'''From tree based algorithm (Random Forest) we got an accuracy score of .84 and the following important features'''


'SW_COM', 'LEN', 'DP_PREP', 'DP_POBJ', 'DP_NLA', 'DP_NNC', 'NER_LOC', 'NER_OTH', 'SNER_ORG',
'POS', 'LEN', 'FG3', 'FG31', 'FG39', 'CG10', 'ND', 'SW_STOP'
       
       
'''fitting the classifier with the above features'''

x1=df[['SW_COM', 'LEN', 'DP_PREP', 'DP_POBJ', 'DP_NLA', 'DP_NNC', 'NER_LOC', 'NER_OTH', 'SNER_ORG',
       'POS', 'LEN', 'FG3', 'FG31', 'FG39', 'CG10', 'ND', 'SW_STOP']]
y1=df['TARGET']

x_train1, x_test1, y_train1, y_test1 = train_test_split(x1, y1, test_size = 0.2, random_state = 0)


rf1=RandomForestClassifier(n_estimators=100)

rf1.fit(x_train1,y_train1)

y_pred1=rf1.predict(x_test1)

accuracy_score(y_pred1, y_test1)

## By using these selected features, model accuracy is also improved to .855 and the model is pretty simplified now.


df['SW_COM'].describe()






























