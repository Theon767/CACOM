# -*- coding: utf-8 -*-
"""
Created on Sun Jul 10 21:45:20 2022

@author: m1380
"""
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.linear_model import LogisticRegression
url =r'C:\Users\m1380\Downloads\train.csv'
train = pd.read_csv(url)
#print(train)
corr=train.corr()
x=train.drop(['y'],axis=1)
y=train['y']
ss=StandardScaler()
x_scaled=ss.fit_transform(x)
model = LogisticRegression()
model.fit(x_scaled, y)
a=model.coef_[0]
print(a)