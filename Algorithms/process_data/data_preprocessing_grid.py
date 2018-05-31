import numpy as np

from Algorithms.process_data import data_preprocessing as preprocess
from Algorithms.process_data import baseline_correction as base_correct


# Function that returns the number of grids within the data, given the grid size of data that we want
def initializeGrid(label,gridlength):

    num_of_grids = 0

    for item in range(len(label)):
        num_of_grids += (label[item].shape[0]//gridlength) * (label[item].shape[1]//gridlength)

    return num_of_grids

# Cut off edges so that we only get complete grid sizes
def cut_off(label,raman,gridlength):

    cut_label = []
    cut_raman = []

    for item in label:
        cut_label.append(item[:(item.shape[0]//gridlength)*gridlength,:(item.shape[1]//gridlength)*gridlength])

    for item in raman:
        cut_raman.append(item[:,:(item.shape[1]//gridlength)*gridlength,:(item.shape[2]//gridlength)*gridlength])

    return cut_label,cut_raman

# Convert the image data into a form where each row contains raman sepctra with one grid, where the
# centre grid is the label to be classified
def convert_grid(data,gridlength,orginial_data,dimensionalized_data,raman_length,row,column_index):

    current_row = row

    # For each raman data we have that have been stretched out into 2D
    for column in range(dimensionalized_data.shape[1]):

        data[row, column_index:column_index + raman_length] = dimensionalized_data[:, column]

        # If you've finished one row of grid shaped data
        if (column + 1) % (gridlength * orginial_data.shape[2]) == 0:
            row += 1
            column_index = 0
            current_row = row

        # If I reach the column length of the image
        elif (column + 1) % orginial_data.shape[2] == 0:
            row = current_row
            column_index += raman_length

        # If I reach the grid length
        elif (column + 1) % gridlength == 0:
            row += 1
            column_index = column_index - ((gridlength-1)*raman_length)

        else:
            column_index += raman_length

    return data,row

# Inserts label for centre classification for each row of data, where each row represents data from a square grid
def obtainLabel(data,label,gridlength):

    # Because the grid will always be square the centres for the row and column will be the same
    first_centre_index = gridlength//2
    counter = 0

    for item in label:
        horz_centre = np.arange(first_centre_index,item.shape[0],gridlength)
        vert_centre = np.arange(first_centre_index,item.shape[1],gridlength)

        for row in horz_centre:
            for column in vert_centre:
                data[counter,-1] = item[row,column]
                counter += 1

    return data

# Main function for obtaining data in the grid format
def obtainNonOverlapGridData(label, raman, gridlength, raman_length):

    row_size = initializeGrid(label, gridlength)

    # Initializing memory to store our input data for out network
    data = np.zeros((row_size, raman_length * (gridlength ** 2) + 1))

    # Cut off excess image
    label, raman = cut_off(label, raman, gridlength)

    # Initializing variable
    varying_row = 0

    # For each image that we have
    for item in raman:

        # Reduce by one dimension
        temp = item.reshape(raman_length,item.shape[1]*item.shape[2])

        # Initialize variable
        row = varying_row
        column_index = 0

        # Converting data to a form where it can be used as an input for the neural network
        data,varying_row = convert_grid(data,gridlength,item,temp,raman_length,row,column_index)

        # Providing the label of the centre grid for

    data = obtainLabel(data,label,gridlength)

    return data

def obtainOverlapGridData(label,raman,gridlength,raman_length):

    # Initializing row size
    row_size = 0

    # For multiple images
    if type(label) is list:

        for item in label:
            row_size += (item.shape[0] - gridlength + 1) * (item.shape[1] - gridlength + 1)

    # For a single image
    else:

        row_size += (label.shape[0] - gridlength + 1) * (label.shape[1] - gridlength + 1)

    data = np.zeros([row_size, raman_length * (gridlength ** 2) + 1])
    row_counter = 0

    # For multiple images
    if type(label) is list:
        for (temp_label,temp_raman) in zip(label,raman):
            for row in range(temp_label.shape[0]-gridlength+1):
                for column in range(temp_label.shape[1]-gridlength+1):

                    column_counter = 0

                    for inner_row in range(gridlength):
                        for inner_column in range(gridlength):

                            data[row_counter,column_counter*raman_length:(column_counter+1)*raman_length] = temp_raman[:,row+inner_row,column+inner_column]
                            column_counter += 1

                            if column_counter == np.ceil(gridlength/2):
                                data[row_counter,-1] = temp_label[row+inner_row,column+inner_column]

                    row_counter += 1

    # For a single image
    else:
        for row in range(label.shape[0] - gridlength + 1):
            for column in range(label.shape[1] - gridlength + 1):

                column_counter = 0

                for inner_row in range(gridlength):
                    for inner_column in range(gridlength):

                        data[row_counter,column_counter*raman_length:(column_counter+1)*raman_length] = raman[:,row+inner_row,column+inner_column]
                        column_counter += 1

                        if column_counter == np.ceil(gridlength/2):
                            data[row_counter, -1] = label[row+inner_row,column+inner_column]

                row_counter += 1

    return data

# Coverts data back to 2D data back to list of 3D data
def revert(raman,data_shape):


    data = []
    counter = 0

    if type(data_shape) is list:
        for item in data_shape:

            num_row = item[0]
            num_column = item[1]

            temp = np.zeros((1024,num_row,num_column))

            for row in range(item[0]):
                for column in range(item[1]):
                    temp[:,row,column] = raman[counter,:]
                    counter += 1

            data.append(temp)

    else:

        num_row = data_shape[0]
        num_column = data_shape[1]

        temp = np.zeros((1024, num_row, num_column))

        for row in range(data_shape[0]):
            for column in range(data_shape[1]):
                temp[:, row, column] = raman[counter, :]
                counter += 1

        data.append(temp)

    return data

