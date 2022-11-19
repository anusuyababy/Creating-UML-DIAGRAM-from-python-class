# -*- coding: utf-8 -*-
"""
Created on Fri Nov 18 15:13:44 2022

@author: kennedy
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 15:54:01 2022

@author: kennedy
"""

import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import operator
import re

#import sweetviz 

from IPython.display import display
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score 
from sklearn.metrics import accuracy_score

class library:
  def __init__(self):
    self.A = head()
  def import_library(self, query, df):
    if operator.contains(query, "import library"):
        print('import pandas as pd') 
        print('import numpy as np') 
        print('import os') 
        print('import plotly.express as px') 
        print('import matplotlib.pyplot as plt') 
        print('import seaborn as sns')
        return
    elif operator.contains(query, "libraries"):
        print('import pandas as pd') 
        print('import numpy as np') 
        print('import os') 
        print('import plotly.express as px') 
        print('import matplotlib.pyplot as plt') 
        print('import seaborn as sns')
        return


class head:
  def __init__(self):
    self.B = EDA()
    self.C = datapreprocessing()
  def show_head(self, query, df):
    if operator.contains(query, "head"):
        s = df.head()
        print(s)

class EDA:
  def show_shape(self, query, df):
    if operator.contains(query, "shape"):
        s = df.shape
        return s
  def show_null(self, query, df):
    if operator.contains(query, "null"):
        s = df.isnull().sum()
        return s
  def correlation(self, query, df):
    if operator.contains(query, "correlation"):
        d = df.corr()
        return d
  def show_column(self, query, df):
    if operator.contains(query, "list the columns"):
        s = df.columns
        return s
  def show_description(self, query, df): 
    if operator.contains(query, "description"):
        d = df.describe()
        return d
  def show_numericalfeat(self, query, df): 
    if operator.contains(query, "numerical feature"):
        featnum=[feature for feature in df.columns if df[feature].dtype !='O']
        s = df[featnum].head(5)
        return s
  def show_categoricalfeat(self, query, df): 
    if operator.contains(query, "categorical feature"):
        featcat=[feature for feature in df.columns if df[feature].dtype =='O']
        s = df[featcat].head(5).to_html()
        return s
    
class datapreprocessing:
  def __init__(self):
    self.D = featureengineering()
  def fill_missingvalues(self, query, df):
      if operator.contains(query, "missing"):
          df.replace('?', np.NaN, inplace=True)
          # seperating categorical feature
          feat_cat=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype =='O']
          # seperating numerical feature
          feat_num=[feature for feature in df.columns if df[feature].isnull().sum()>0 and df[feature].dtype !='O']
          # taking median for numerical features
          s = df[feat_num].median()
          df[feat_num]=df[feat_num].fillna(s)
          # filling the missing values with "missing" for categorical features
          df[feat_cat]=df[feat_cat].fillna('Missing')
          s = df.head()
          return s

class featureengineering:
  def __init__(self):
    self.E = traintestsplit()
  def label_encoding(self, query, df):   
      if operator.contains(query, "label encoding"):
          le = LabelEncoder()
          objList = df.select_dtypes(include = "object").columns
          print (objList)
          for feat in objList:
              df[feat] = le.fit_transform(df[feat].astype(str))
          s = df.head()
          return s
      
class traintestsplit:
  def __init__(self):
    self.F = modelbuilding()
  def traintest_split(self, query, df): 
     if operator.contains(query, "split the data"):
         n = int(''.join(filter(str.isdigit, query)))
         s = int(str(n)[-2:])
         d = s/100
         x = df.iloc[:,:-1]
         y = df.iloc[: , -1:]
         X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=d, random_state=0)
         d = X_train.shape, X_test.shape
         return d 
     
class modelbuilding:
  def naivebayes_classifier(self, query, df): 
      if operator.contains(query, "naive bayes"):
          le = LabelEncoder()
          objList = df.select_dtypes(include = "object").columns
          print (objList)
          for feat in objList:
              df[feat] = le.fit_transform(df[feat].astype(str))
          x = df.iloc[:,:-1]
          y = df.iloc[: , -1:]
          x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
          # Gaussian naive bayes
          clf = GaussianNB()
          clf.fit(x_train, y_train)
          NBscore = clf.score(x_train, y_train)
          y_pred=clf.predict(x_test)
          p = accuracy_score(y_test,y_pred)
          p1 = classification_report(y_test, y_pred)
          p2 = confusion_matrix(y_test, y_pred)
          p3 = precision_score(y_test, y_pred, average="micro")
          p4 = recall_score(y_test, y_pred, average='micro')
          p5 = f1_score(y_test, y_pred)
          report = pd.DataFrame()
          report['Model'] = ['Gaussian Naive bayes'] 
          report['Training Accuracy'] = [NBscore] 
          report['Test Accuracy'] = [p] 
          report['Confusion matrix'] = [p2] 
          report['Precision'] = [p3] 
          report['Recall'] = [p4] 
          report['F1-score'] = [p5] 
          z = report.to_html()
          return z
  def randomforest_classifier(self, query, df): 
      if operator.contains(query, "forest"):
          le = LabelEncoder()
          objList = df.select_dtypes(include = "object").columns
          print (objList)
          for feat in objList:
              df[feat] = le.fit_transform(df[feat].astype(str))
          x = df.iloc[:,:-1]
          y = df.iloc[: , -1:]
          x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
          # random forest
          clf1 = RandomForestClassifier(n_estimators=20)
          clf1.fit(x_train, y_train)
          rfscore = clf1.score(x_train, y_train)
          y_pred1=clf1.predict(x_test)
          q = accuracy_score(y_test,y_pred1)
          q1 = classification_report(y_test, y_pred1)
          q2 = confusion_matrix(y_test, y_pred1)
          q3 = precision_score(y_test, y_pred1, average="micro")
          q4 = recall_score(y_test, y_pred1, average="micro")
          q5 = f1_score(y_test, y_pred1)
          report = pd.DataFrame()
          report['Model'] = ['Random Forest']
          report['Training Accuracy'] =[rfscore] 
          report['Test Accuracy'] = [q]
          report['Confusion matrix'] = [q2]
          report['Precision'] = [q3] 
          report['Recall'] = [q4] 
          report['F1-score'] = [q5] 
          z = report.to_html()
          return z
  def decisiontree_classifier(self, query, df):
      if operator.contains(query, "tree"):
          le = LabelEncoder()
          objList = df.select_dtypes(include = "object").columns
          print (objList)
          for feat in objList:
              df[feat] = le.fit_transform(df[feat].astype(str))
          x = df.iloc[:,:-1]
          y = df.iloc[: , -1:]
          x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
          # Decision tree classifier
          clf3 = DecisionTreeClassifier()
          clf3.fit(x_train, y_train)
          dtscore = clf3.score(x_train, y_train)
          y_pred3=clf3.predict(x_test)
          s = accuracy_score(y_test,y_pred3)
          s1 = classification_report(y_test, y_pred3)
          s2 = confusion_matrix(y_test, y_pred3)
          s3 = precision_score(y_test, y_pred3, average="micro")
          s4 = recall_score(y_test, y_pred3, average='micro')
          s5 = f1_score(y_test, y_pred3)
          report = pd.DataFrame()
          report['Model'] = ['Decission Tree'] 
          report['Training Accuracy'] = [dtscore] 
          report['Test Accuracy'] = [s]
          report['Confusion matrix'] = [s2]
          report['Precision'] = [s3]
          report['Recall'] = [s4]
          report['F1-score'] = [s5]
          z = report.to_html()
          return z
  def knn_classifier(self, query, df):
      if operator.contains(query, "KNN"):
          le = LabelEncoder()
          objList = df.select_dtypes(include = "object").columns
          print (objList)
          for feat in objList:
              df[feat] = le.fit_transform(df[feat].astype(str))
          x = df.iloc[:,:-1]
          y = df.iloc[: , -1:]
          x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
          # KNN CLASSIFIER
          clf2 = KNeighborsClassifier()
          clf2.fit(x_train, y_train)
          knnscore = clf2.score(x_train, y_train)
          y_pred2=clf2.predict(x_test)
          r = accuracy_score(y_test,y_pred2)
          r1 = classification_report(y_test, y_pred2)
          r2 = confusion_matrix(y_test, y_pred2)
          r3 = precision_score(y_test, y_pred2, average="micro")
          r4 = recall_score(y_test, y_pred2, average='micro')
          r5 = f1_score(y_test, y_pred2)
          report = pd.DataFrame()
          report['Model'] = ['KNN'] 
          report['Training Accuracy'] =[knnscore] 
          report['Test Accuracy'] = [r]
          report['Confusion matrix'] = [r2]
          report['Precision'] = [r3]
          report['Recall'] = [r4]
          report['F1-score'] = [r5]
          z = report.to_html()
          return z
  def SVM_classifier(self, query, df):
      if operator.contains(query, "svm"):
          le = LabelEncoder()
          objList = df.select_dtypes(include = "object").columns
          print (objList)
          for feat in objList:
              df[feat] = le.fit_transform(df[feat].astype(str))
          x = df.iloc[:,:-1]
          y = df.iloc[: , -1:]
          x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
          # SVM Classifier
          clf4 = svm.SVC()
          clf4.fit(x_train, y_train)
          svmscore = clf4.score(x_train, y_train)
          y_pred4=clf4.predict(x_test)
          t = accuracy_score(y_test,y_pred4)
          t1 = classification_report(y_test, y_pred4)
          t2 = confusion_matrix(y_test, y_pred4)
          t3 = precision_score(y_test, y_pred4, average="micro")
          t4 = recall_score(y_test, y_pred4, average='micro')
          t5 = f1_score(y_test, y_pred4)
          report = pd.DataFrame()
          report['Model'] = ['SVM']
          report['Training Accuracy'] =[svmscore]
          report['Test Accuracy'] = [t]
          report['Confusion matrix'] = [t2]
          report['Precision'] = [t3]
          report['Recall'] = [t4]
          report['F1-score'] = [t5]
          z = report.to_html()
          return z
  def adaboost_classifier(self, query, df):
      if operator.contains(query, "adaboost"):
          le = LabelEncoder()
          objList = df.select_dtypes(include = "object").columns
          print (objList)
          for feat in objList:
              df[feat] = le.fit_transform(df[feat].astype(str))
          x = df.iloc[:,:-1]
          y = df.iloc[: , -1:]
          x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
          # adaboost classifier
          clf5 = AdaBoostClassifier()
          clf5.fit(x_train, y_train)
          adascore = clf5.score(x_train, y_train)
          y_pred5=clf5.predict(x_test)
          u = accuracy_score(y_test,y_pred5)
          u1 = classification_report(y_test, y_pred5)
          u2 = confusion_matrix(y_test, y_pred5)
          u3 = precision_score(y_test, y_pred5, average="micro")
          u4 = recall_score(y_test, y_pred5, average='micro')
          u5 = f1_score(y_test, y_pred5)
          report = pd.DataFrame()
          report['Model'] = ['Adaboost']
          report['Training Accuracy'] =[adascore]
          report['Test Accuracy'] = [u]
          report['Confusion matrix'] = [u2]
          report['Precision'] = [u3]
          report['Recall'] = [u4]
          report['F1-score'] = [u5]
          z = report.to_html()
          return z
  def gradientboosting_classifier(self, query, df):
      if operator.contains(query, "gradient boosting"):
          le = LabelEncoder()
          objList = df.select_dtypes(include = "object").columns
          print (objList)
          for feat in objList:
              df[feat] = le.fit_transform(df[feat].astype(str))
          x = df.iloc[:,:-1]
          y = df.iloc[: , -1:]
          x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=0)
          # gradient boosting
          clf6 = GradientBoostingClassifier()
          clf6.fit(x_train, y_train)
          gradientscore = clf6.score(x_train, y_train)
          y_pred6=clf6.predict(x_test)
          v = accuracy_score(y_test,y_pred6)
          v1 = classification_report(y_test, y_pred6)
          v2 = confusion_matrix(y_test, y_pred6)
          v3 = precision_score(y_test, y_pred6, average="micro")
          v4 = recall_score(y_test, y_pred6, average='micro')
          v5 = f1_score(y_test, y_pred6)
          report = pd.DataFrame()
          report['Model'] = ['Gradient boosting']
          report['Training Accuracy'] =[gradientscore]
          report['Test Accuracy'] = [v]
          report['Confusion matrix'] = [v2]
          report['Precision'] = [v3]
          report['Recall'] = [v4]
          report['F1-score'] = [v5]
          z = report.to_html()
          return z
  

text = "import library"
df=pd.read_csv("D:/professor work/voice to code converter/voice to python code converter/Voice-to-Python-code-converter-main/salary.csv")

# Driver code
# Object instantiation
#Rodger = library()
#r = library() 
#s = shape()
#d = shape()
# Accessing class attributes
# and method through objects
#print(Rodger.attr1)
l = library()

