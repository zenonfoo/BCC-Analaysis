## Data Extraction ##

import numpy as np
import h5py

# Finding Specific Tissue
def findSpecificTissue(folder_name,tissue,testing_label):

    data = h5py.File(folder_name + tissue + '.mat')
    label_map = np.array(data[testing_label])
    RamanMap = np.array(data['map_t' + tissue])

    return label_map,RamanMap

# Finding the unique keys for all the tissue files -  to identify callable keys when pre-processing data
def findUniqueKeys(folder_name):

    unique_keys = []

    for x in range(3,72):

        # For the tissues with only a single task
        try:
            mat = h5py.File(folder_name + str(x) + '.mat')

            for item in list(mat.keys()):
                unique_keys.append(item)

        except OSError:
            print(x, 'OSError')

        except KeyError:
            print(x, 'KeyError')

        # For the tissues with multiple task
        try:

            for y in range(1,3):
                mat = h5py.File(folder_name + str(x) + '.mat')

                for item in list(mat.keys()):
                    unique_keys.append(item)

        except OSError:
            print(x, y, 'OSError')

        except KeyError:
            print(x, y, 'KeyError')

    return np.unique(unique_keys)

# Given an input key, this function will return the tissues which contain that key
def findTissue(folder_name,key):

    tissue = []

    for x in range(3,72):

        try:
            mat = h5py.File(folder_name + str(x) + '.mat')

            if key in list(mat.keys()):
                tissue.append()

        except OSError:
            print(x, 'OSError')

        except KeyError:
            print(x, 'KeyError')

    return tissue

# This function converts the ones in the matrix to the index - to be one hot encoded later
def convert(matrix,index):

    for x in range(matrix.shape[0]):
        for y in range(matrix.shape[1]):
            if matrix[x,y] == 1:
                matrix[x,y] = index

    return matrix

# Returns 3 lists - Map of labels indicated , Raman Data, Tissues Used
# Map of label - in the form of a list of 2D matrices
# Raman Data - in the form of a list of 3D matrices
def preProcessAllLabels(folder_name):

    # Initializing variables
    known_labels = ['bcc','dermis','fat','epi','dye']
    label = []
    RamanData = []
    tissues_used = []

    for x in range(3, 72):

        for y in range(0,3):

            if y == 0:
                folder_number = str(x)

            else:
                folder_number = str(x) + '_' + str(y)

            try:
                # Opening file and obtaining keys
                mat = h5py.File(folder_name + folder_number + '.mat')
                keys = list(mat.keys())

                # Checking just for aforementioned keys in all the files
                present_labels = [item for item in keys if item in known_labels]

                # Only if bcc is within the image do we take the rest of the data
                if 'bcc' in present_labels:

                    label_map = np.zeros(mat[present_labels[0]].shape)

                    for key in present_labels:

                        if key == 'bcc':

                            temp = convert(np.array(mat[key]), 1)
                            label_map = label_map + temp

                        elif key == 'dermis':

                            temp = convert(np.array(mat[key]),2)
                            label_map = label_map + temp

                        elif key == 'fat':

                            temp = convert(np.array(mat[key]),3)
                            label_map = label_map + temp

                        elif key == 'epi':

                            temp = convert(np.array(mat[key]),4)
                            label_map = label_map + temp

                        elif key == 'dye':

                            temp = convert(np.array(mat[key]),5)
                            label_map = label_map + temp

                    RamanMap = np.array(mat['map_t' + folder_number])

                else:
                    continue

                # Cheking for dimension mismatch
                # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
                if label_map.shape[0] == RamanMap.shape[1] and label_map.shape[1] == RamanMap.shape[2]:
                    label.append(label_map)
                    RamanData.append(RamanMap)
                    tissues_used.append(folder_number)

            except OSError:
                print(x, OSError)

            # This is due to the tissue not having any of the keys
            except KeyError:
                print(x, KeyError)

            except IndexError:
                print(x, IndexError)

            # This error is due to different keys having different shapes
            except ValueError:
                print(x, ValueError)

    return label, RamanData, tissues_used


# This is for the tissues with multiple tasks
# Returns 3 lists - Map of Label, Raman Data, Tissues Used
# Map of Label - in the form of a list of 2D matrices
# Raman Data - in the form of a list of 3D matrices
def preProcessBCC(folder_name,testing_label):

    # Initializing variables
    label = []
    RamanData = []
    tissues_used = []

    for x in range(3, 72):

        for y in range(0,3):

            if y == 0:
                folder_number = str(x)

            else:
                folder_number = str(x) + '_' + str(y)

            try:
                # Opening file and obtaining keys
                mat = h5py.File(folder_name + folder_number +'.mat')
                keys = list(mat.keys())

                # Checking just for BCC data in all the files
                if testing_label in keys:
                    label_map = np.array(mat[testing_label])
                    RamanMap = np.array(mat['map_t' + folder_number])

                else:
                    continue

                # Cheking for dimension mismatch
                # If there is no dimension mismatch organise data to be raman input and BCC/NonBCC cell output
                if label_map.shape[0] == RamanMap.shape[1] and label_map.shape[1] == RamanMap.shape[2]:
                    label.append(label_map)
                    RamanData.append(RamanMap)
                    tissues_used.append(folder_number)

            except OSError:
                print(x, 'OSError')

            except KeyError:
                print(x, 'KeyError')

            except IndexError:
                print(x, 'IndexError')

    return label, RamanData, tissues_used