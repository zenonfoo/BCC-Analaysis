# Built in libraries
import numpy as np
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
folder = './Data/RamanData/tissue_'
label = 'bcc'
cv_set = [(0,3),(3,6),(6,9),(9,12),(12,16)]

# Loading Data
label_data,raman_data,tissues_used = obtain.preProcessBCC(folder_name=folder,testing_label=label)

for cv in cv_set:

    # Spltting into Training and Testing Set
    print("Splitting data into training and testing set")

    training_label = []
    testing_label = []
    training_label.append(label_data[:cv[0]])
    training_label.append(label_data[cv[1]:])
    testing_label.append(label_data[cv[0]:cv[1]])

    training_raman = []
    testing_raman = []
    training_raman.append(raman_data[:cv[0]])
    training_raman.append(raman_data[cv[1]:])
    testing_raman.append(raman_data[cv[0]:cv[1]])

    test_tissue = tissues_used[cv[0]:cv[1]]

    train_set,train_shapes = preprocess.organiseData(training_label,training_raman)
    test_set,test_shapes = preprocess.organiseData(testing_label,testing_raman)

    X_train = train_set[:,:-1]
    y_train = train_set[:,-1]
    X_test = test_set[:,:-1]
    y_test = test_set[:,-1]

    # Clearing memory
    del train_set
    del test_set
    del train_shapes
    del test_shapes
    del training_raman
    del testing_raman
    del training_label
    del testing_label

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

    # Clearing memory
    del X_train
    del X_test

    # Initializing parameters for neural network
    parameters = training.create_paramaters(input_dim=100,units=500,layers=4,initializer='uniform',
                                            validation_split=0,activation='sigmoid',output_activation='sigmoid',
                                            optimizer='adam',batch=5000,epochs=50)

    # Training Neural Network
    classifier = training.neural_network(X_train_pca,y_train,parameters)

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
        plt.colorbar()


    bone()
    multiplier = 40000

    for num, tissue in enumerate(test_tissue):
        image = classifier.predict(X_test_pca[num * multiplier:(num + 1) * multiplier, :])
        make_image(image)
        plt.title('Tissue ' + tissue)
        plt.savefig('Tissue ' + tissue + ' Cross Validation')
        plt.close()






