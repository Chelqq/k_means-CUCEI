# -*- coding: utf-8 -*-
"""
Created on Mon Oct 31 12:21:42 2022

@author: Xel
"""

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Wholesale_customers_data.csv')

#know data shape
data.shape

#is there any null val?
data.isnull().sum()

#value types
data.dtypes

###Sample values
###random sample values are selected to verify cluster belonging
indexes = [16,176, 392]
samples = pd.DataFrame(data.loc[indexes],
                       columns = data.keys()).reset_index(drop = True)
data = data.drop(indexes, axis = 0)

#we dont need region and channel columns, drop in samples and training
data = data.drop(['Region', 'Channel'], axis=1)
samples = samples.drop(['Region', 'Channel'], axis=1)

#escalating data samples and training
from sklearn import preprocessing 
data_escalated = preprocessing.Normalizer().fit_transform(data)
samples_escalated = preprocessing.Normalizer().fit_transform(samples)



############################## ML ANALYSIS
from sklearn.cluster import KMeans

#Evaluated var
X = data_escalated.copy()

###Obtainig the best val 4 K
###I will be using the -->elbow method<-- to find K
###Then calculate the clustering algo given different val for X
inertia = []
for i in range(1,20):
    algo = KMeans(n_clusters=i, init='k-means++',
                  max_iter=300, n_init=10)
    algo.fit(X)
    ###For each K, we calculate the sum of total squared val in the Cluster
    inertia.append(algo.inertia_)

#sum of total squared error
plt.figure(figsize=[10,6])
plt.title('Elbow method')
plt.xlabel()


