## Training Neural Network fo classifying BCC and non BCC cells ##
## Contains core functions of to preprocess data such as normalizing and splitting ##

import numpy as np

folder_name = 'BCC&NoBCC_Classification/2/BCC_Data_2.npy'

# Importing dataset
def import_data(folder_name):

    data = np.load(folder_name)
    X = data[:,:-1]
    y = data[:,-1]

    return data,X,y

# Feature Scaling
def normalize(X_train,X_test):

    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train,X_test,sc

# Constructing eigenvalues and eigenvectors
def eigen(X_train):

    cov_mat = np.cov(X_train.T)
    eigen_vals, eigen_vectors = np.linalg.eig(cov_mat)

    return eigen_vals, eigen_vectors

# Constructing History class to store data on accuracy and losses as the network trains
import keras

class History(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

# Creating parameters to be used as argument during calling of the training function
def create_paramaters(input_dim,units,layers,initializer,validation_split,activation,output_activation,optimizer,batch,epochs):

    parameters = {}
    parameters['input_dimension'] = input_dim
    parameters['units'] = units
    parameters['layers'] = layers
    parameters['initializer'] = initializer
    parameters['validation_split'] = validation_split
    parameters['activation'] = activation
    parameters['output_activation'] = output_activation
    parameters['optimizer'] = optimizer
    parameters['batch'] = batch
    parameters['epochs'] = epochs

    return parameters

# Running ANN algorithm
def neural_network(X_train, y_train, parameters):

    # Making the ANN
    from keras.models import Sequential
    from keras.layers import Dense

    # Initializing the ANN
    classifier = Sequential()

    # Adding input and hidden layers
    for layer in range(parameters['layers']):

        if layer == 0:
            # kernel_initializer refers to inital weights
            classifier.add(Dense(units=parameters['units'], kernel_initializer=parameters['initializer'],
                                 activation=parameters['activation'], input_dim=parameters['input_dimension']))

        else:
            classifier.add(Dense(units=parameters['units'], kernel_initializer=parameters['initializer'],
                                 activation=parameters['activation']))

    # Adding output layer
    classifier.add(Dense(units=1, kernel_initializer=parameters['initializer'],
                         activation=parameters['activation']))

    # Initializing classes to store loss and accuracy history
    # history = History()

    # Compiling the ANN
    # optimizer: algorithm you want to use to find the optimal set of weights, adam is stochastic gradient descent
    # loss: loss function - binary_crossentropy is the logarithmic loss function
    classifier.compile(optimizer=parameters['optimizer'], loss='binary_crossentropy', metrics=['accuracy'])

    # Fitting classifier to training set
    classifier.fit(X_train, y_train, batch_size=parameters['batch'], epochs=parameters['epochs'])

    return classifier