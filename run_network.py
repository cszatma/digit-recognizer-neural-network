import mnist_loader
import network

# Load images
training_data, validation_data, test_data = mnist_loader.load_data_wrapper()

# Setup network with 1 hidden layer of 30 neurons
net = network.Network([784, 30, 10])

# Use stochastic gradient descent to learn from the MNIST training data over 30 epochs, with a mini-batch size of 10
# and a learning rate of η = 3
net.sgd(training_data, 30, 10, 3.0, test_data=test_data)

# Another test using 100 neurons in the hidden layer
net = network.Network([784, 100, 10])
net.sgd(training_data, 30, 10, 3.0, test_data=test_data)