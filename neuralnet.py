from math import exp
import numpy as np
import utils

class Layer:
    def __init__(self, neurons, prev_neurons, prev_layer, succ_layer, \
            activation, activation_derivative):

        self.previous = prev_layer
        self.successive = succ_layer
        self.weights = np.random.random((neurons, prev_neurons + 1)) - 0.5
        self.activation = activation
        self.activation_derivative = activation_derivative

    def value(self):
        # don't forget the bias term!
        prev = np.vstack([1, self.previous.value()])
        self.v = self.weights.dot(prev)
        self.y = self.activation(self.v)
        return self.y

class InputLayer(Layer):
    def __init__(self, neurons):
        Layer.__init__(self, neurons, 0, None, None, None, None)

        self.val = []
        self.weights = None

    def setValue(self, inputs):
        self.val = np.array([inputs]).T
        self.y = self.val
        self.v = self.val

    def value(self):
        return self.val

class NeuralNetwork:
    def __init__(self, ns, activation='sigmoid', activation_derivative=None):
        self.layers = []
        if type(activation) is str:
            activation, activation_derivative = utils.activations[activation]

        self.activation = np.vectorize(activation, otypes = [np.float])
        self.activation_derivative = np.vectorize(activation_derivative, \
            otypes = [np.float])

        # layer creation
        for i in range(len(ns)):
            if i == 0:
                self.layers.append(InputLayer(ns[i]))
            else:
                self.layers.append(Layer(ns[i], ns[i - 1], None, None, \
                    self.activation, self.activation_derivative))

        # layer linking
        for i in range(len(ns)):
            if i > 0:
                self.layers[i].previous = self.layers[i - 1]

            if i < len(ns) - 1:
                self.layers[i].successive = self.layers[i + 1]

    def value(self, inputs):
        self.layers[0].setValue(inputs)
        return list(self.layers[-1].value().T[0])

    def backprop(self, inputs, wanted_outputs, learning_rate):
        self.layers[0].setValue(inputs)
        out = self.layers[-1].value()
        error = 0.5 * np.square(wanted_outputs - out).sum()

        # transform into column vectors
        inputs = np.array([inputs]).T
        wanted_outputs = np.array([wanted_outputs]).T

        # find output layer deltas
        deltas = ((wanted_outputs - out) * self.activation_derivative(
            self.layers[-1].v))

        for i in range(len(self.layers) - 1, 0, -1):
            # previous layer's values, including the bias
            prev_y = np.vstack([1, self.layers[i - 1].y])
            prev_v = np.vstack([1, self.layers[i - 1].v])

            # compute gradient and update weights
            gradient = deltas * prev_y.T
            self.layers[i].weights += gradient * learning_rate

            # calculate deltas for the previous layer
            a = self.activation_derivative(prev_v)
            b = self.layers[i].weights.T.dot(deltas)

            # don't need delta for the bias unit
            deltas = (a * b)[1:] 

        return error
