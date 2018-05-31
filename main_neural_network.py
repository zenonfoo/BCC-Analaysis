# Built in libraries
import numpy as np
from pylab import bone
import matplotlib.pyplot as plt

# Personal libraries
from Algorithms.process_data import obtaining_data as obtain
from Algorithms.process_data import data_preprocessing_grid as grid_process
from Algorithms.process_data import data_preprocessing as preprocess
from Algorithms.neural_network import assess
from Algorithms.neural_network import training
from Algorithms.process_data import baseline_correction as base_correct


### Preprocessing Data ###
# Initializing Variables
folder = './Data/RamanData/tissue_'
label = 'bcc'

# Loading Data
label_data,raman_data,tissues_used = obtain.preProcessBCC(folder_name=folder,testing_label=label)

# Loading All Data
label_data,raman_data,tissues_used = obtain.preProcessAllLabels(folder)

### Training ###
# Loading Data if data is already saved
print('Loading Data')
folder_name = 'BCC&NoBCC_Classification/4/BCC_Data_4.npy'
data,X,y = training.import_data(folder_name)

# Splitting dataset into training and test set
print('Splitting Data')
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.1,stratify=y)

# Feature Scaling
# sc variable is to be used later on to fit testing data
print('Normalizing Data')
X_train,X_test,sc = training.normalize(X_train,X_test)

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






