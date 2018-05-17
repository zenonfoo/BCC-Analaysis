import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import trapz

from Algorithms.neural_network import assess as testing
from Algorithms.neural_network import training as training
from Algorithms.process_data import data_preprocessing as preprocess

### Investigating different learning rate decays ###

### Preprocessing Data ###
# Initializing Variables
folder = './RamanData/tissue_'
label = 'bcc'

# Loading Data
label_data,raman_data,tissues_used = preprocess.preProcessBCC(folder_name=folder,testing_label=label)

# Processing Data
data = preprocess.organiseData(label=label_data,raman=raman_data)

### Training ###
# Separating input and output data
X = data[:, :-1]
y = data[:, -1]

# Loading Data if data is already saved
print('Loading Data')
folder_name = 'BCC&NoBCC_Classification/2/BCC_Data_2.npy'
data,X,y = training.import_data(folder_name)

# Splitting dataset into training and test set
print('Splitting Data')
X_train, X_test, y_train, y_test = training.split(X,y)

# Feature Scaling
# sc variable is to be used later on to fit testing data
print('Normalizing Data')
X_train,X_test,sc = training.normalize(X_train,X_test)

# Initializing different layers
layers = np.arange(1,5)

# Initializing different units
units = np.arange(100,1000,100)

# Initializing different epochs
epochs = np.arange(5,50,5)

# Initializing variable to store ROC Data
ROC_Data = []

# Initializing variable to store history data
history_data = []

# Trying different parameters
 # sgd = optimizers.SGD(lr=rate)

for epoch in epochs:

# Initializing parameters for neural network
    parameters = training.create_paramaters(input_dim=1024,units=100,layers=3,initializer='uniform',
                                        validation_split=0,activation='sigmoid',output_activation='sigmoid',
                                        optimizer='adam',batch=1000, epochs=epoch)

    # Training neural network
    classifier, history = training.neural_network(X_train,y_train,parameters)

    history_data.append(history)

    ### Testing ###
    # ROC Curve
    # Initializing thresholds
    thresholds = np.arange(0.1, 1, 0.02)

    # Generating ROC data
    roc = testing.ROC(classifier, X_test, y_test, thresholds)

    # Generating ROC column
    # roc['Units'] = [unit]*len(thresholds)
    # roc['Layers'] = [layer]*len(thresholds)
    roc['Epochs'] = [epoch]*len(thresholds)

    if len(ROC_Data) == 0:
        ROC_Data = roc
    else:
        ROC_Data = pd.concat([ROC_Data,roc],axis=0)


# Converting ROC_Data into dictionary with normalized Area Under ROC Data
thresh = len(thresholds)
area = []
for item in range(len(layers)*len(units)):

    info = {}
    begin = item*thresh
    end = (item+1)*thresh
    FPR = ROC_Data['FPR'].iloc[begin:end]
    TPR = ROC_Data['TPR'].iloc[begin:end]
    FPR = FPR[::-1]
    TPR = TPR[::-1]
    info['layers'] = ROC_Data['Layers'].iloc[begin]
    info['units'] = ROC_Data['Units'].iloc[begin]
    info['area'] = trapz(TPR,x=FPR)/(max(TPR)*max(FPR))
    info['accuracy'] = history_data[item].acc
    info['losses'] = history_data[item].losses
    area.append(info)

# Plotting scatter plot of units, layers and normalized area under ROC
x=[]
y=[]
z=[]
for item in area:

    x.append(item['layers'])
    y.append(item['units'])
    z.append(item['area'])

plt.scatter(x=x,y=y,c=z)

# Plotting ROC
thresh = len(thresholds)
area = []

for item in range(len(epochs)):

    info = {}
    begin = item*thresh
    end = (item+1)*thresh

    if ROC_Data['Epochs'].iloc[begin] == 10:
        FPR = ROC_Data['FPR'].iloc[begin:end]
        TPR = ROC_Data['TPR'].iloc[begin:end]
        plt.plot(FPR,TPR,label='Epochs ' + str(epochs[item]))

plt.close()
plt.close()
plt.close()
plt.close()

fig, ax = plt.subplots()
counter = 1
for label, df in temp.groupby('layers'):
    df.plot('units', 'area', ax=ax, label='Layers: ' + str(counter))
    counter += 1
plt.legend()


# Converting ROC_Data into dictionary with normalized Area Under ROC Data
thresh = len(thresholds)
info = {}
info['layers'] = []
info['units'] = []
info['area'] = []
for item in range(len(layers)*len(units)):

    begin = item*thresh
    end = (item+1)*thresh
    FPR = temp['FPR'].iloc[begin:end]
    TPR = temp['TPR'].iloc[begin:end]
    FPR = FPR[::-1]
    TPR = TPR[::-1]
    info['layers'].append(temp['Layers'].iloc[begin])
    info['units'].append(temp['Units'].iloc[begin])
    info['area'].append(trapz(TPR,x=FPR)/(max(TPR)*max(FPR)))