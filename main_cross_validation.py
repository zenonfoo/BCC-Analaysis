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
from Algorithms.process_data import data_preprocessing_grid as grid_process

### Preprocessing Data ###
# Initializing Variables
folder = './Data/RamanData/tissue_'
label = 'bcc'
cv_set = [(0,3),(3,6),(6,9),(9,12),(12,16)]
AUC = []

for cv in cv_set:

    # Loading Data
    label_data, raman_data, tissues_used = obtain.preProcessBCC(folder_name=folder, testing_label=label)

    # Spltting into Training and Testing Set
    print("Splitting data into training and testing set")

    training_label = label_data[:cv[0]] + label_data[cv[1]:]
    testing_label = label_data[cv[0]:cv[1]]

    training_raman = raman_data[:cv[0]] + raman_data[cv[1]:]
    testing_raman = raman_data[cv[0]:cv[1]]

    test_tissue = tissues_used[cv[0]:cv[1]]

    train_set,train_shapes = preprocess.organiseData(training_label,training_raman)
    test_set,test_shapes = preprocess.organiseData(testing_label,testing_raman)

    X_train = train_set[:,:-1]
    X_test = test_set[:,:-1]
    y_train = train_set[:,-1]
    y_test = test_set[:,-1]

    # Clearing memory
    del label_data
    del raman_data
    del train_set
    del test_set
    del training_raman
    del testing_raman
    del tissues_used

    # Baseline Correction
    print("Baseline Correction")
    X_train = base_correct.polynomial(X_train,2)
    X_test = base_correct.polynomial(X_test,2)

    # Feature Scaling
    print("Feature Scaling")
    X_train,X_test,sc = training.normalize(X_train,X_test)

    del sc

    # PCA
    print("Performing PCA")
    num_pca = 100
    pca = PCA()
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)

    del X_train

    X_train_pca = X_train_pca[:,:num_pca]
    X_test_pca = pca.transform(X_test)

    del X_test
    X_test_pca = X_test_pca[:,:num_pca]

    # Clearing memory
    del pca

    # Grid processing
    print("Grid Preprocessing")
    grid_length = 3
    X_train_pca = grid_process.revert(X_train_pca, train_shapes)
    X_test_pca = grid_process.revert(X_test_pca, test_shapes)
    X_train_pca = grid_process.obtainOverlapGridData(training_label, X_train_pca, grid_length, num_pca)
    X_test_pca = grid_process.obtainOverlapGridData(testing_label, X_test_pca, grid_length, num_pca)
    y_train = X_train_pca[:, -1]
    X_train_pca = X_train_pca[:, :-1]
    y_test = X_test_pca[:,-1]
    X_test_pca = X_test_pca[:, :-1]

    # Clearing memory
    del training_label
    del testing_label
    del test_shapes
    del train_shapes

    # Initializing parameters for neural network
    parameters = training.create_paramaters(input_dim=900,units=450,layers=4,initializer='uniform',
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
    roc = assess.ROC(classifier,X_test_pca,y_test,thresholds)

    # Calculating AUC
    # AUC.append(np.trapz(roc.TPR, x=roc.FPR) / (max(roc.TPR) * max(roc.FPR)))

    # Storing roc data
    AUC.append(roc)

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
    multiplier = int(X_train_pca.shape[0]/(16-(cv[1]-cv[0])))

    for num, tissue in enumerate(test_tissue):
        image = classifier.predict(X_test_pca[num * multiplier:(num + 1) * multiplier, :])
        make_image(image)
        plt.title('Tissue ' + tissue)
        plt.savefig('Tissue ' + tissue + ' Cross Validation 3x3 Grid')
        plt.close()

    # Clearing memory
    del classifier
    del parameters
    # del image
    del roc
    del thresholds
    del X_train_pca
    del X_test_pca
    del y_train
    del y_test