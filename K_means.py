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



"""
Sample values
random sample values are selected to verify cluster belonging
"""
indexes = [16,176, 392]
samples = pd.DataFrame(data.loc[indexes],
                       columns = data.keys()).reset_index(drop = True)
data = data.drop(indexes, axis = 0)

#we dont need region and channel columns, drop in samples and training
data = data.drop(['Region', 'Channel'], axis=1)
samples = samples.drop(['Region', 'Channel'], axis=1)

from sklearn import preprocessing
data.escalated = preprocessing.Normalizer().fit_transform(data)
