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

    def backpropagate(self, deltas, learning_rate):
        # previous layer's values, including the bias
        prev_y = np.vstack([1, self.previous.y])
        prev_v = np.vstack([1, self.previous.v])

        # compute gradient and update weights
        gradient = deltas * prev_y.T
        self.weights += gradient * learning_rate

        # calculate deltas for the previous layer
        a = self.activation_derivative(prev_v)
        b = self.weights.T.dot(deltas)

        # don't need delta for the bias unit
        deltas = (a * b)[1:]

        self.previous.backpropagate(deltas, learning_rate)


class InputLayer(Layer):
    def __init__(self, *args, **kwargs):
        Layer.__init__(self, *args, **kwargs)

        self.val = []
        self.weights = None
        self.previous = None

    def setValue(self, inputs):
        if isinstance(inputs, np.ndarray):
            self.val = inputs.T
        else:
            self.val = np.array([inputs]).T
        self.y = self.val
        self.v = self.val

    def value(self):
        return self.val

    def backpropagate(self, deltas, learning_rate):
        pass


class OutputLayer(Layer):
    def backpropagate(self, wanted_outputs, learning_rate):
        deltas = ((wanted_outputs - self.y) * self.activation_derivative(self.v))
        Layer.backpropagate(self, deltas, learning_rate)


class NeuralNetwork:
    def __init__(self, ns, activation='sigmoid', activation_derivative=None):
        if type(activation) is str:
            activation, activation_derivative = utils.activations[activation]

        self.activation = np.vectorize(activation, otypes = [np.float])
        self.activation_derivative = np.vectorize(activation_derivative, \
            otypes = [np.float])

        # layer creation
        self.layers = []
        for i in range(len(ns)):
            if i == 0:
                layer_type = InputLayer
            elif i == len(ns) - 1:
                layer_type = OutputLayer
            else:
                layer_type = Layer

            self.layers.append(layer_type(ns[i], ns[i - 1], None, None, \
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

    def backprop(self, wanted_outputs, learning_rate):
        error = 0.5 * np.square(wanted_outputs - self.layers[-1].y).sum()

        # transform into column vector and backpropagate
        wanted_outputs = np.array([wanted_outputs]).T
        self.layers[-1].backpropagate(wanted_outputs, learning_rate)

        return error

    def output_derivatives(self):
        """
        calculate the derivatives of the outputs with respect to the inputs

        returns a numpy NxM array, where N is the number of output neurons and M
        is the number of input neurons, such that the element at row i and
        column j is the derivative of the i-th output neuron with respect to
        the j-th input neuron
        """
        prev_layer = None
        for i in range(len(self.layers) - 1, 0, -1):
            layer = self.layers[i]

            this_layer = layer.weights * np.concatenate(
                [self.activation_derivative(layer.v)] * layer.weights.shape[1],
                axis = 1)

            prev_layer = (prev_layer.dot(this_layer) if prev_layer is not None
                else this_layer)[:, 1:]

        return prev_layer
