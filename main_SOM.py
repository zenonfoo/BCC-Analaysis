import numpy as np
import pandas as pd
from Algorithms.self_organizing_map import minisom

from Algorithms.neural_network import training as training

# Loading Data
print('Loading Data')
data,X,y = training.import_data('BCC&NoBCC_Classification/4/BCC_Data_4.npy')
del data

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
print('Feature Scaling')
sc = MinMaxScaler()
X = sc.fit_transform(X)

# Initializing the size of the grid for the SOM and the number features in the dataset
som = minisom(x=2,y=3,input_len=1024)

# Randomly initializing the weights of each node
som.random_weights_init(X)

# Training
print('Training')
som.train_random(data=X,num_iteration=5000)

### Visualizing the results ###
from pylab import bone,pcolor,colorbar

# Creates Figure
bone()

# Plots colormap - the transpose is so that it is in the right form for the pcolor function to be compared later
# when adding the markers etc
pcolor(som.distance_map().T)
colorbar()

### Plotting ####

# Obtain coordinate data for all data
def obtain_winning_node(som,X):

    all_plots = []
    winner = []

    for i in range(len(X)):

        winner.append(som.winner(X[i,:]))

        if (i + 1) % 40000 == 0:
            # Converting unique coordinate to unique value
            winner = pd.factorize(winner)
            winner = winner[0]

            all_plots.append(winner)

            winner = []

    return all_plots

# Converting Winning Node
tissues_used = np.load('BCC&NoBCC_Classification/4/Tissues_Used_Data_4.npy')
def organise_winning_node(image_data,tissue_name):

    altered_data = {}

    for item,name in zip(image_data,tissue_name):

        # Reshaping value into 2D image
        image = item.reshape(200,200)

        # Rotating image to match orginal
        image = np.rot90(image)

        # Storing in dict
        altered_data[name] = image

    return altered_data

