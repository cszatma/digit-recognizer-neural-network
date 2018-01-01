# Used to load the MNIST image data

import pickle
import gzip
import numpy as np


def load_data():
    # Return the MNIST data as a tuple containing the training data, the validation data, and the test data
    # The training data is returned as a tuple with two entries
    # The first entry contains the actual training images. This is a numpy ndarray with 50,000 entries
    # Each Entry is a numpy ndarray with 784 values, representing 28 * 28  784 pixels in a single MNIST image
    # The second entry in the tuple is a numpy ndarray containing 50,000 entries which are the digit values (0...9)
    # The validation and test data are similar, except each contains 10,000 images
    file = gzip.open('mnist.pkl.gz', 'rb')
    training_data, validation_data, test_data = pickle.load(file, encoding="latin1")
    file.close()
    return training_data, validation_data, test_data


def load_data_wrapper():
    # Return a tuple containing (training_data, validation_data, test_data)
    # Based on load_data() but the format is more convenient for use in the neural network
    # In particular, training_data is a list containing 50,000 binary-tuples (x, y)
    # x is a 784-dimensional numpy.ndarray containing the input image
    # y is a 10-dimensional numpy.ndarray representing the unit vector corresponding to the correct digit for x
    # validation_data and test_data are lists containing 10,000 binary-tuples (x, y)
    # In each case, x is a 784-dimensional numpy.ndarray containing the input image
    # y is the corresponding classification, i.e., the digit values corresponding to x
    tr_d, va_d, te_d = load_data()
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)

    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])

    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])

    return training_data, validation_data, test_data


def vectorized_result(j):
    # Returns a 10-dimensional unit vector with a 1.0 in the jth positions and zeroes everywhere else
    # This is used to convert a digit (0...9) into a corresponding desired output from the neural network
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
