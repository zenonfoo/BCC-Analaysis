# Built in libraries
import numpy as np
import pandas as pd
from pylab import bone
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Personal libraries
from Algorithms.process_data import obtaining_data as obtain
from Algorithms.process_data import data_preprocessing as preprocess
from Algorithms.neural_network import assess
from Algorithms.neural_network import training
from Algorithms.process_data import baseline_correction as base_correct

### Preprocessing Data ###
# Initializing Variables
folder = './RamanData/Raw_Data/tissue_'
label = 'bcc'

# Loading Data
label_data,raman_data,tissues_used = obtain.preProcessBCC(folder_name=folder,testing_label=label)

# Spltting into Training and Testing Set
print("Splitting data into training and testing set")
no_testing = 2
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
del no_testing
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
num_pca = 100
pca = PCA()
pca.fit(X_train)
X_train_pca = pca.transform(X_train)
X_train_pca = X_train_pca[:,:num_pca]
X_test_pca = pca.transform(X_test)
X_test_pca = X_test_pca[:,:num_pca]

# Initializing different layers
layers = np.arange(1,6)

# Initializing different units
units = np.arange(10,110,10)

# Initializing variable to store ROC Data
ROC_Data = {}

# Initializing thresholds for calculating ROC
thresholds = np.arange(0.1, 1, 0.02)

# Trying different parameters
 # sgd = optimizers.SGD(lr=rate)

for layer in layers:

    ROC_Data[layer] = []

    for unit in units:

        # Initializing parameters for neural network
            parameters = training.create_paramaters(input_dim=100,units=unit,layers=layer,initializer='uniform',
                                                validation_split=0,activation='sigmoid',output_activation='sigmoid',
                                                optimizer='adam',batch=5000,epochs=50)

            # Training neural network
            classifier = training.neural_network(X_train_pca,y_train,parameters)

            ### Testing ###
            # ROC Curve
            # Generating ROC data - returns a dataframe with columns = [TPR,FPR,Thresholds]
            roc = assess.ROC(classifier, X_test_pca, y_test, thresholds)

            # Storing ROC Data
            ROC_Data[layer].append(roc)

# Initializing DataFrame to store AUC Data - columns:layers, rows:units
AUC = np.zeros((len(layers),len(units)))
AUC = pd.DataFrame(AUC)
AUC.columns = units

# Converting ROC_Data into dictionary with normalized Area Under ROC Data
thresh = len(thresholds)
row_counter = 0

for key in ROC_Data:

    column_counter = 0

    for item in ROC_Data[key]:

        FPR = item['FPR']
        TPR = item['TPR']
        FPR = FPR[::-1]
        TPR = TPR[::-1]
        AUC.iloc[row_counter,column_counter] = np.trapz(TPR, x=FPR) / (max(TPR) * max(FPR))
        column_counter += 1

    row_counter += 1

# Plotting AUC
for item in range(5):

    num_layer = item + 1
    plt.plot(AUC.iloc[item],label='Layers ' + str(num_layer))

plt.legend()
plt.xlabel('Number of Nodes in Each Layer')
plt.ylabel('AUC')
plt.grid()
plt.title('Varying Network Parameters For PCA 100 Input')
plt.savefig('Varying Network Parameters For PCA 100 Input')