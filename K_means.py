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

#showing and knowing sum of total squared error
plt.figure(figsize=[10,6])
plt.title('Elbow method')
plt.xlabel('Cluster No.')
plt.ylabel('Inertia')
plt.plot(list(range(1,20)), inertia, marker='x')
plt.show()

###Aplying clustering Algo##
###We define the Algorithm with K val
algo = KMeans(n_clusters=6, init='k-means++',
              max_iter=400, n_init=10)

##Tranning....
algo.fit(X)

##Centroid data is obtained, so labels
centroids, labels = algo.cluster_centers_, algo.labels_

#Using sample data to verify to wich cluster it belong
sample_prediction = algo.predict(samples_escalated)
for i, pred in enumerate(sample_prediction):
    print("Sample", i, "Found in cluster:", pred)
    

### Show results next to data ###
# Dimensionality reduction to data
from sklearn.decomposition import PCA

model_pca = PCA(n_components = 2)
model_pca.fit(X)
pca = model_pca.transform(X) 

#We reduce dimensionality to centroids
centroids_pca = model_pca.transform(centroids)

#Pretty colors 4 each cluster
colors = ['blue', 'red', 'green', 'orange', 'gray', 'brown']
colors_cluster = [colors[labels[i]] for i in range(len(pca))]

#ilustrating PCA components
plt.scatter(pca[:, 0], pca[:, 1], c = colors_cluster, 
            marker = 'o',alpha = 0.4)

#Showing centroids
plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1],
            marker = 'x', s = 100, linewidths = 3, c = colors)

#keeping vars with easy names
xvector = model_pca.components_[0] * max(pca[:,0])
yvector = model_pca.components_[1] * max(pca[:,1])
columns = data.columns

#Ilustrating cluster names with vector distance
for i in range(len(columns)):
    #Showing Vectors
    plt.arrow(0, 0, xvector[i], yvector[i], color = 'black', 
              width = 0.0005, head_width = 0.02, alpha = 0.75)
    #putting names
    plt.text(xvector[i], yvector[i], list(columns)[i], color='black', 
             alpha=0.75)

plt.show()