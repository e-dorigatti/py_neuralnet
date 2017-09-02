from math import exp, sqrt
import numpy as np
from .errors import common as common_errors
from .activations import common as common_activations
import itertools


def _to_numpy_column(l):
    if isinstance(l, np.ndarray):
        ret = l.T if l.shape[1] != 1 else l
    else:
        ret = np.array([l]).T

    assert isinstance(ret, np.ndarray) and len(ret.shape) == 2 and ret.shape[1] == 1
    return ret


class NeuralNetwork:
    def __init__(self, ns, activation='sigmoid', error='quadratic'):
        self.activation = common_activations.get(activation, activation)
        self.error = common_errors.get(error, error)
        self.ns = ns
        
        self.v = [ np.zeros((n, 1)) for n in ns ]
        self.y = [ np.zeros((n, 1)) for n in ns ]

        # black magic random initialization range 
        self.weights = [ sqrt(24.0 / (n1 + n2)) * (np.random.random((n1, n2 + 1)) - 0.5)
            for n1, n2 in zip(ns[1:], ns[:-1]) ]

    def value(self, inputs):
        """
        Calculates the output value of the neural network using the given inputs.

        Inputs can be either python lists or numpy row/column arrays
        """
        inputs = _to_numpy_column(inputs)

        self.y[0] = self.v[0] = inputs
        for i, weights in enumerate(self.weights):
            prev = np.vstack([1, self.y[i]])
            self.v[i + 1] = weights.dot(prev)
            self.y[i + 1] = self.activation.f(self.v[i + 1])

        return self.y[-1]

    def prediction_error(self, wanted_outputs, actual_outputs):
        wanted_outputs = _to_numpy_column(wanted_outputs)
        actual_outputs = _to_numpy_column(actual_outputs)

        return self.error.error(wanted_outputs, actual_outputs)


    def calculate_gradients(self, wanted_outputs, actual_outputs):
        """
        Calculates the partial derivatives of the error with respect to the weights
        given the actual outputs of the network and the desired outputs.

        Returns a list whose each element is a numpy array with the same shape as
        the corresponding weight matrix.
        """
        wanted_outputs = _to_numpy_column(wanted_outputs)
        actual_outputs = _to_numpy_column(actual_outputs)

        a = self.error.deltas(wanted_outputs, actual_outputs)
        b = self.activation.f_prime(self.v[-1])
        deltas = (a * b)

        gradients = list()
        for weights, y, v in reversed(zip(self.weights, self.y[:-1], self.v[:-1])):
            prev_y = np.vstack([1, y])
            prev_v = np.vstack([1, v])

            gradients.append(deltas * prev_y.T)
            assert gradients[-1].shape == weights.shape

            a = self.activation.f_prime(prev_v)
            b = weights.T.dot(deltas)
            deltas = (a * b)[1:]

        self._input_deltas = deltas
        return reversed(gradients)

    def update_weights(self, gradients, learning_rate):
        """
        Update the weights according to the gradient descent algorithm using
        the given gradient (as computed by calculate_gradients) and the given
        learning rate.
        """
        for weights, gradient in zip(self.weights, gradients):
            assert gradient.shape == weights.shape
            weights += gradient * learning_rate;

    def backpropagate(self, wanted_outputs, actual_outputs, learning_rate):
        """
        Update the weights of the network in order to reduce the prediction error.
        """
        actual_outputs = _to_numpy_column(actual_outputs)
        wanted_outputs = _to_numpy_column(wanted_outputs)

        gradients = self.calculate_gradients(wanted_outputs, actual_outputs)
        self.update_weights(gradients, learning_rate)

        return self.prediction_error(wanted_outputs, actual_outputs)

    def output_derivatives(self):
        """
        calculate the derivatives of the outputs with respect to the inputs

        returns a numpy NxM array, where N is the number of output neurons and M
        is the number of input neurons, such that the element at row i and
        column j is the derivative of the i-th output neuron with respect to
        the j-th input neuron
        """
        prev_layer = None
        for values, weights in reversed(zip(self.v[1:], self.weights)):
            a = np.concatenate([self.activation.f_prime(values)] * weights.shape[1], axis=1)
            this_layer = weights * a

            prev_layer = (prev_layer.dot(this_layer) if prev_layer is not None
                else this_layer)[:, 1:]

        assert prev_layer.shape == (self.ns[-1], self.ns[0])
        return prev_layer
