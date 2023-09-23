import random

import numpy as np

# Here, object corresponds to a class that Network inherits from
class Network(object):
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        # TYPE: an array of arrays
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        # TYPE: an array of matrices
        self.weights = [np.random.randn(sizes[y], sizes[y-1])
                        for y in range(1, self.num_layers)]

    # a - the activations passed into the input layer (of size size[0])
    def feedforward(self, a):
        a_last = a
        for i in range(sizes-1):
           z = np.matmul(self.weights[i], a_last) + self.biases[i]
           a_last = sigmoid(z)
        return a_last

    """
    Purpose: Trains neural network using mini-batch stochastic gradient descent. 
             If "test_data" is provided then network is evaluated against test data
                after every epoch, and partial progress is printed out (NOTE: printing 
                partial progress is a SLOW operation)

    Data: "training data" is a list of tuples "(x, y)," representing training input (x)
            and desired output (y)
    """
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        # Note: control blocks like if and while don't scope variables in Python
        if test_data: 
            n_test = len(test_data)
        n = len(training_data)
        # For every epoch
        for j in range(epochs):
            random.shuffle(training_data)
            # Compute the mini_batches ahead of time
            mini_batches = [
                    training_data[k:k+mini_batch_size]
                    for k in range(0, n, mini_batch_size)]
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            if test_data:
                print(f"Epoch {j}: {self.evaluate(test_data)} / {n_test}") 
            else: 
                print(f"Epoch {j} complete") 

        """
        # Attempt 1: 
        # Abbreviations
        mbs = mini_batch_size

        # For every epoch
        for i in range(epochs)
            # randomly select mbs size of data from training_data
            # For each spoch
            random.shuffle(training_data)
            for offset in range(0, len(training_data), mbs):
                for i in range(offset, offset + mbs):
                    # approxqimate the gradient function
        """

    """
    Purpose: update entre network's weights and biases by applying gradient
             descent using backprogation to a single mini batch. 
    Data: "mini_batch" is a list of tuples "(x, y)"
    """
    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb+dnb for nb, dnb in list(zip(nabla_b, delta_nabla_b))]
            nabla_w = [nw+dnw for nw, dnw in list(zip(nabla_w, delta_nabla_w))]
        self.weights = [w-(eta/len(mini_batch))*nw 
                        for w, nw in list(zip(self.weights, nabla_w))]
        self.biases = [b-(eta/len(mini_batch))*nb
                        for b, nb in list(zip(self.biases, nabla_b))]

    """
    Purpose: returns a tuple representing gradient for cost function
    Data: -> (nabla_b, nabla_w), where nabla_b and nabla_w are layer-by-layer 
             lists of numpy arrays
    """
    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        """feedforward""" 
        activations = x
        """list storing all activations, layer by layer"""
        activations = [x] 
        """list storing all z vectors, layer-by-layer"""
        zs = []           
        for b, w in list(zip(self.biases, self.weights)):
            z = np.dot(w, activation)+b
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        """ backward pass """
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        """ Note: in the following code, l is the lth last layer. 
            Examples: 
            l = 1 means the last layer of neurons 
            l = 2 means the second-last layer"""
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)


    """
    Purpose: return # of test inputs for which neural network outputs
             correct result. Note: neural network's output is assumed to 
             be index of whichever neuron in final layer has highest activation
    """
    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), y)
            for (x, y) in test_data]

        """ Note: int(x == y) casts from a boolean to an int: 0 or 1 """         
        return sum(int(x == y) for (x,y) in test_results)

    """
    Purpose: return vector of partial derivatives \partial C_x / \partial a 
             for the output activations
    """
    def cost_derivative(self, output_activations, y):
        return (output_activations-y)


### Miscellaneous functions
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
