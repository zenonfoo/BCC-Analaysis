### Data Preprocessing for BCC and non BCC ###

import numpy as np

from Algorithms.process_data import obtaining_data


# Looks at the dimensions of each tissue data to initialize matrix to store them in one variable
def initalizeMatrix(label):

    row_size = 0

    if type(label) is list:
        for item in range(len(label)):
            row_size += label[item].shape[0] * label[item].shape[1]

    # This is the condition that there is only one matrix
    else:
        row_size = label.shape[0] * label.shape[1]

    return row_size

# Converts all the tissue data into a singular 2D matrix variable to be ready for the neural network
def organiseData(label,raman):

    row_size = initalizeMatrix(label)
    data = np.zeros((row_size,1025))
    data_shape = []
    counter = 0

    if type(label) is list:

        for item in range(len(label)):
            data_shape.append(label[item].shape)
            for row in range(label[item].shape[0]):
                for column in range(label[item].shape[1]):

                    data[counter,:-1] = raman[item][:, row, column]
                    data[counter,-1] = label[item][row, column]
                    counter += 1

    else:

        data_shape = label.shape
        for row in range(label.shape[0]):
            for column in range(label.shape[1]):
                data[counter, :-1] = raman[:, row, column]
                data[counter, -1] = label[row, column]
                counter += 1

    return data,data_shape

# Separating data - with the main goal of normalizing the dataset w.r.t only the non-bcc raman data
def separateBCC(data):

    # Number of rows in data
    row_data = data.shape[0]

    # Counting and summing number of bcc elements
    count_bcc = [1 for item in range(row_data) if data[item,-1] == 1]
    sum_bcc = sum(count_bcc)

    # Initializing store for bcc and non bcc data
    bcc_store = np.empty([sum_bcc,1025])
    non_bcc_store = np.empty([row_data-sum_bcc,1025])

    # bcc and nonbcc counters
    bcc_counter = 0
    non_bcc_counter = 0

    # Separating Data
    for item in range(row_data):

        if data[item,-1] == 1:
            bcc_store[bcc_counter,:] = data[item,:]
            bcc_counter += 1

        else:
            non_bcc_store[non_bcc_counter,:] = data[item,:]
            non_bcc_counter += 1

    return bcc_store[:,:-1],non_bcc_store[:,:-1]

def separateAll(data):

    # Number of rows in data
    row_data = data.shape[0]

    # Finding number of labels
    no_labels = int(max(data[:,-1]) + 1)

    # Initializing counter for all labels
    counter = np.zeros((1,no_labels))

    # Counting labels
    for item in range(row_data):
        counter[0,int(data[item,-1])] += 1

    # Initializing memory to store different label
    label_store = {}

    for item in range(no_labels):
        label_store[item] = np.zeros((int(counter[0,item]),1024))

    # Separating Data
    label_counter = np.zeros((1,no_labels))

    for item in range(row_data):
        label_store[int(data[item,-1])][int(label_counter[0,int(data[item,-1])]),:] = data[item,:-1]
        label_counter[0, int(data[item,-1])] += 1

    return label_store


# Counts the number of each label within the dataset
def count(labels):

    label_counter = {}

    for item in labels:
        if item not in label_counter:
            label_counter[item] = 1

        else:
            label_counter[item] += 1

    return label_counter

# Returns BCC data in 2D matrix
def obtain_BCC_data(filename):

    label,raman,tissue = obtaining_data.preProcessBCC(filename, 'bcc')
    (data,data_shape) = organiseData(label,raman)

    return data,tissue

# Organise Data
def organise_BCC_data(data,index,tissue_used):

    # Obtaining testing data
    testing_data = data[index*40000:index*40000+40000,:]

    # Obtaining training data
    training_index = np.ones(data.shape[0],dtype=bool)
    training_index[index*40000:index*40000+40000] = False
    training_data = data[training_index,:]

    # Obtaining testing tissue
    testing_tissue = tissue_used[index]


    return training_data,testing_data,testing_tissue




