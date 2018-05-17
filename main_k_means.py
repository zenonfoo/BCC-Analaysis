### K Means Algorithm ###
import numpy as np
from sklearn.cluster import KMeans
from Algorithms.process_data import baseline_correction as base_correct

# Importing Dataset
print('Loading Data')
data = np.load('BCC&NoBCC_Classification/4/BCC_Data_4.npy')
X = data[:,:-1]
del data

# Baseline Correction
print('Baseline Correction')
X = base_correct.polynomial(X,2)

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
print('Feature Scaling')
sc = MinMaxScaler()
X = sc.fit_transform(X)

# Applying kmean
print('Applying Kmeans')
kmeans = KMeans(n_clusters=6,max_iter=700).fit(X)

