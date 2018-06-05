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
del train_shapes
del test_shapes

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

# Initializing parameters for neural network
parameters = training.create_paramaters(input_dim=9216,units=500,layers=4,initializer='uniform',
                                        validation_split=0,activation='sigmoid',output_activation='sigmoid',
                                        optimizer='adam',batch=5000,epochs=30)

# Training Neural Network
classifier,history = training.neural_network(X[:,:-1],X[:,-1],parameters)

### Testing ###
# ROC Curve
# Initializing thresholds
thresholds = np.arange(0.1,1,0.02)

# Generating ROC data
print('Generating ROC')
roc = assess.ROC(classifier,X_test,y_test,thresholds)

### Plotting probability distribution as labelled image

# Obtaining Image
def make_image(image_data):

    axis_length = int(len(image_data)**0.5)
    image = image_data.reshape(axis_length,axis_length)
    image = np.rot90(image)
    plt.pcolor(image)
    plt.axis('image')
    plt.axis('off')
    # plt.colorbar()


bone()

for item,tissue in zip(label_data,tissues_used):

    image = np.rot90(item)
    plt.pcolor(image)
    plt.axis('off')
    plt.axis('image')
    plt.title('Tissue ' + tissue)
    plt.savefig('Tissue ' + tissue)
    plt.close()






