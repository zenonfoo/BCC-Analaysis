## Polynomial Baseline Correction ##
import numpy as np

# Returns polynomial fitted datapoints
def polynomial(data,order):

    # Row length and column length of data
    row_length = data.shape[0]
    col_length = data.shape[1]

    # Initializing memory to store coefficients for each raman spectra
    coeff = np.zeros((row_length,order+1))

    # Initializing x datapoints
    x = np.arange(1024)

    # Initializing memory to store fitted datapoints
    datapoints = np.zeros((row_length,col_length))

    # Obtaining coefficients
    for item in range(row_length):

        coeff[item,:] = np.polyfit(x,data[item,:],order)

    # Calculating datapoints
    for item in range(col_length):

        coefficients = np.zeros((row_length,order+1))

        for var in range(order+1):

            coefficients[:,var] = coeff[:,var] * (item ** (order-var))

        datapoints[:,item] = np.sum(coefficients,axis=1)

    datapoints = data - datapoints

    return datapoints