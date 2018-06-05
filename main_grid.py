# Built in libraries
import numpy as np
from pylab import bone
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Personal libraries
from Algorithms.process_data import obtaining_data as obtain
from Algorithms.process_data import data_preprocessing as preprocess
from Algorithms.process_data import data_preprocessing_grid as grid_process
from Algorithms.neural_network import training
from Algorithms.process_data import baseline_correction as base_correct

### Preprocessing Data ###
# Initializing Variables
folder = './Data/RamanData/tissue_'
label = 'bcc'

# Loading Data
label_data,raman_data,tissues_used = obtain.preProcessBCC(folder_name=folder,testing_label=label)

# Spltting into Training and Testing Set
print("Splitting data into training and testing set")
no_testing = 4
train_set,train_shapes = preprocess.organiseData(label_data[:-no_testing],raman_data[:-no_testing])
test_set,test_shapes = preprocess.organiseData(label_data[-no_testing:],raman_data[-no_testing:])

label_data_train = label_data[:-no_testing]
label_data_test = label_data[-no_testing:]
X_train = train_set[:,:-1]
y_train = train_set[:,-1]
X_test = test_set[:,:-1]
y_test = test_set[:,-1]
test_tissue = tissues_used[-no_testing:]

# Clearing memory
del train_set
del test_set
del raman_data
del label_data

# Baseline Correction
print("Baseline Correction")
X_train = base_correct.polynomial(X_train,2)
X_test = base_correct.polynomial(X_test,2)

# Feature Scaling
print("Feature Scaling")
X_train,X_test,sc = training.normalize(X_train,X_test)

# PCA
print("Performing PCA")
num_pca = 50
pca = PCA()
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_train_pca = X_train_pca[:,:num_pca]
X_test_pca = pca.transform(X_test)
X_test_pca = X_test_pca[:,:num_pca]

# Clearing memory
del X_train
del X_test

# Grid processing
print("Grid Preprocessing")
grid_length = 9
X_train_pca = grid_process.revert(X_train_pca,train_shapes)
X_test_pca = grid_process.revert(X_test_pca,test_shapes)
X_train_pca = grid_process.obtainOverlapGridData(label_data_train,X_train_pca,grid_length,num_pca)
X_test_pca = grid_process.obtainOverlapGridData(label_data_test,X_test_pca,grid_length,num_pca)
y_train = X_train_pca[:,-1]
X_train_pca = X_train_pca[:,:-1]
X_test_pca = X_test_pca[:,:-1]

# Initializing parameters for neural network
input_dimension = int(X_train_pca.shape[1])
layer_size = int(input_dimension/2)
parameters = training.create_paramaters(input_dim=input_dimension,units=layer_size,layers=4,initializer='uniform',
                                        validation_split=0,activation='sigmoid',output_activation='sigmoid',
                                        optimizer='adam',batch=5000,epochs=50)

# Training Neural Network
classifier = training.neural_network(X_train_pca,y_train,parameters)

### Plotting probability distribution as labelled image

# Obtaining Image
def make_image(image_data):

    axis_length = int(len(image_data)**0.5)
    image = image_data.reshape(axis_length,axis_length)
    image = np.rot90(image)
    plt.pcolor(image)
    plt.axis('image')
    plt.axis('off')
    plt.colorbar()

bone()
multiplier = int(X_train_pca.shape[0]/12)

for num,tissue in enumerate(test_tissue):

    image = classifier.predict(X_test_pca[num*multiplier:(num+1)*multiplier,:])
    make_image(image)
    plt.title('Tissue ' + tissue)
    plt.savefig('Tissue ' + tissue + ' PCA50 9x9 Grid')
    plt.close()




