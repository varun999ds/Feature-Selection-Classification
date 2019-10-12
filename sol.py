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

'''Checking the null records if any'''
df.isnull().sum()

'''Checking for duplicate records'''
df.duplicated().sum()
### there are 3119 duplicate records, removing them from the dataset
df=df.drop_duplicates()

'''Cheching for duplicate features'''
df.columns.duplicated()
### No duplicate column

'''Checking for low/no variance features'''
from sklearn.feature_selection import VarianceThreshold
select=VarianceThreshold(threshold=0)
### No feature with 0 variance


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
        
'''Applying ANOVA gave us following features with high f-statistic score:
('LEN', 521.838789327431)
('DP_PREP', 541.7514610994156)
('DP_POBJ', 555.1416886378594)
('CG3', 711.2671288132983)
('CG8', 952.705645926629)
('CG9', 801.9525001579449)
('CG10', 1188.1559409324057)
('CG14', 619.3005494316915)
('CG23', 958.4878352253514)
('CG29', 654.4618785946835)
('CG33', 506.57117193232983)
('CG34', 986.5893839601983)
('CG35', 614.7642256092269)
('CG38', 593.1692695505961)
('CG47', 586.4241284613535)
('CG49', 518.8937741171383)
('C_TITLE', 1714.3686658671263)
('C_KEYW', 579.2435093629758)'''

'''Applying Logistic regression, LASSO Regression to get the feature importance and performance of regression model for the dataset'''
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.preprocessing import StandardScaler

x=df.drop(['TARGET', 'LABEL'], axis=1)
y=df['TARGET']

sc=StandardScaler()
xscaled=sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(xscaled, y, test_size = 0.2, random_state = 0)

lr=LogisticRegression(multi_class='auto')
lr.fit(x_train,y_train)
predlog=lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(predlog,y_test)
accuracy_score(predlog,y_test)
### getting an accuracy of .78

'''Using lasso regression to get the features with coefficient >0'''
a=[]
sel_ = SelectFromModel(LogisticRegression(C=1, penalty='l1'))
sel_.fit(x_train,y_train,)


for feature_list_index in sel_.get_support(indices=True):
    a.append(x.columns[feature_list_index])
    print(x.columns[feature_list_index])

'''following features does not have 0 coefficient in Lasso penalty
['FG1', 'FG2', 'FG3', 'FG4', 'FG5', 'FG6', 'FG7', 'FG8', 'FG9', 'FG10', 'FG11', 'FG12', 'FG13', 'FG14', 'FG15', 'FG16', 'FG17', 'FG18', 'FG19', 'FG20', 'FG21', 'FG22', 'FG23', 'FG24', 'FG25', 'FG26', 'FG27', 'FG28', 'FG29', 'FG30', 'FG31', 'FG32', 'FG33', 'FG34', 'FG35', 'FG36', 'FG37', 'FG38', 'FG39', 'FG40', 'FG41', 'FG42', 'FG43', 'FG44', 'FG45', 'FG46', 'FG47', 'FG48', 'FG49', 'FG50', 'SW_COM', 'SW_SEMCOL', 'SW_STOP', 'SW_COL', 'ND', 'LEN', 'DP_DOBJ', 'DP_PREP', 'DP_POBJ', 'DP_NSUBJ', 'DP_NLA', 'DP_NRA', 'DP_NNC', 'CG8', 'CG10', 'CG12', 'CG18', 'CG19', 'CG23', 'CG24', 'CG28', 'CG29', 'CG31', 'CG33', 'CG34', 'CG38', 'CG39', 'CG41', 'CG42', 'CG45', 'CG47', 'CG49', 'C_TITLE', 'C_AUTH', 'C_AFF', 'C_HIGL', 'C_CORR', 'C_GRANT', 'C_FUND', 'NER_PER', 'NER_ORG', 'NER_LOC', 'NER_OTH', 'SNER_ORG', 'SNER_PST', 'SNER_ADD', 'SNER_CITY', 'SNER_NAME', 'POS', 'SEP']'''




'''Fitting logistic regression with only these features'''

x=df[['FG1', 'FG2', 'FG3', 'FG4', 'FG5', 'FG6', 'FG7', 'FG8', 'FG9', 'FG10', 'FG11', 'FG12', 'FG13', 'FG14', 'FG15', 'FG16', 'FG17', 'FG18', 'FG19', 'FG20', 'FG21', 'FG22', 'FG23', 'FG24', 'FG25', 'FG26', 'FG27', 'FG28', 'FG29', 'FG30', 'FG31', 'FG32', 'FG33', 'FG34', 'FG35', 'FG36', 'FG37', 'FG38', 'FG39', 'FG40', 'FG41', 'FG42', 'FG43', 'FG44', 'FG45', 'FG46', 'FG47', 'FG48', 'FG49', 'FG50', 'SW_COM', 'SW_SEMCOL', 'SW_STOP', 'SW_COL', 'ND', 'LEN', 'DP_DOBJ', 'DP_PREP', 'DP_POBJ', 'DP_NSUBJ', 'DP_NLA', 'DP_NRA', 'DP_NNC', 'CG8', 'CG10', 'CG12', 'CG18', 'CG19', 'CG23', 'CG24', 'CG28', 'CG29', 'CG31', 'CG33', 'CG34', 'CG38', 'CG39', 'CG41', 'CG42', 'CG45', 'CG47', 'CG49', 'C_TITLE', 'C_AUTH', 'C_AFF', 'C_HIGL', 'C_CORR', 'C_GRANT', 'C_FUND', 'NER_PER', 'NER_ORG', 'NER_LOC', 'NER_OTH', 'SNER_ORG', 'SNER_PST', 'SNER_ADD', 'SNER_CITY', 'SNER_NAME', 'POS', 'SEP']]

sc=StandardScaler()
xscaled=sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(xscaled, y, test_size = 0.2, random_state = 0)

lr=LogisticRegression(multi_class='auto')
lr.fit(x_train,y_train)
predlog=lr.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix

confusion_matrix(predlog,y_test)
accuracy_score(predlog,y_test)

### getting an accuracy of .77 on the validation data, but this model is more simple and there are low chances of overfitting












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

## since the rf classifier is giving an accuracy score of .83, checking for the features significant in the classification
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






























