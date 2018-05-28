# -*- coding: utf-8 -*-
"""
Created on Sun Mar 25 14:16:49 2018

@author: lenovo
"""

import sys
import numpy
import pandas as pd
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans


def KMeansClustering(data,tot_sum_of_squares): 

    print("\nK-Means Clustering:\n")
    for k in range(2,11):
        kmeans = KMeans(n_clusters=k, random_state = 0)
        kmeans.fit(data)
        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_
        y = numpy.zeros(k)
        for i in range(0,(data.shape[0])):
            y[labels[i]] += sum(((data[i,:]) - (centroids[labels[i],:]))**2)
        
        tot_within_sum_of_squares = sum(y)
        
        ratio = tot_within_sum_of_squares/tot_sum_of_squares
        
        print("Ratio using ",k," cluster(s): ", ratio)
        
def HClustering(data,tot_sum_of_squares): 

    print("\nH-Clustering:\n")
    for k in range(2,11):
        model = AgglomerativeClustering(n_clusters=k)
        model.fit(data)
        labels = model.labels_
        
        centroids = numpy.zeros((k,data.shape[1]))
        count = numpy.zeros(k)
        
        for i in range(0,(data.shape[0])):
            centroids[labels[i],:] += data[i,:]
            count[labels[i]] += 1
            
        for j in range(0,k):
            centroids[j,:] /= count[j];
 
        y = numpy.zeros(k)
        for i in range(0,(data.shape[0])-1):
            y[labels[i]] += sum(((data[i,:]) - (centroids[labels[i],:]))**2)
        
        tot_within_sum_of_squares = sum(y)
        
        ratio = tot_within_sum_of_squares/tot_sum_of_squares
        
        print("Ratio using ",k," cluster(s): ", ratio)
    
def GaussianClustering(data,tot_sum_of_squares): 

    print("\nGaussian Mixture Models:\n")
    for k in range(2,11):
        gmm = GaussianMixture(n_components=k).fit(data)
        labels = gmm.predict(data)
        
        centroids = numpy.zeros((k,data.shape[1]))
        count = numpy.zeros(k)
        
        for i in range(0,(data.shape[0])):
            centroids[labels[i],:] += data[i,:]
            count[labels[i]] += 1
            
        for j in range(0,k):
            centroids[j,:] /= count[j];
 
        y = numpy.zeros(k)
        for i in range(0,(data.shape[0])-1):
            y[labels[i]] += sum(((data[i,:]) - (centroids[labels[i],:]))**2)
        
        tot_within_sum_of_squares = sum(y)
        
        ratio = tot_within_sum_of_squares/tot_sum_of_squares
        
        print("Ratio using ",k," cluster(s): ", ratio)

#######################################################################################
 
print("\n-------------------------------------------------------------------\n")       
print("\nAnalysis of dataset_Facebook.csv:\n")
    
# reading dataset 
data = pd.read_csv(sys.argv[1], delimiter =';')

#converting to numerical values
le = preprocessing.LabelEncoder()
data = data.apply(le.fit_transform)

# scaling the data
scaler = StandardScaler()
scaler.fit(data)
data = scaler.transform(data)


# finding average of data points component wise
avg_var = data.sum(axis=0)
avg_var = pd.Series(avg_var).divide(data.shape[0])

tot_sum_of_squares = 0

for i in range(0,(data.shape[0])):
    tot_sum_of_squares += sum(((data[i,:]) - (pd.Series(avg_var).values))**2)
    
KMeansClustering(data,tot_sum_of_squares)
HClustering(data,tot_sum_of_squares)
GaussianClustering(data,tot_sum_of_squares)
    
########################################################################################

print("\n-------------------------------------------------------------------\n") 
print("\nAnalysis of Frogs_MFCCs.csv:\n")
# reading dataset 
frog_data = pd.read_csv(sys.argv[2], delimiter =',')
frog_data = frog_data.iloc[:,1:22]
frog_data= frog_data.values

# finding average of data points component wise
avg_var = frog_data.sum(axis=0)
avg_var = pd.Series(avg_var).divide(frog_data.shape[0])

tot_sum_of_squares = 0

for i in range(0,(frog_data.shape[0])):
    tot_sum_of_squares += sum(((frog_data[i,:]) - (pd.Series(avg_var).values))**2)
    
KMeansClustering(frog_data,tot_sum_of_squares)
HClustering(frog_data,tot_sum_of_squares)
GaussianClustering(frog_data,tot_sum_of_squares)


