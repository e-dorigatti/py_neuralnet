from itertools import izip
import numpy as np
import random


def minibatch(nnet, input_set, batch_size, learning_rate, regularization_coeff, stop):
    if not hasattr(learning_rate, '__call__'):
        x = float(learning_rate)
        learning_rate = lambda _: x

    if not hasattr(stop, '__call__'):
        n = stop
        stop = lambda i, e, l: i > n

    epoch, error, lrate = 0, 1000.0, 0
    while not stop(epoch, error, lrate):
        epoch += 1
        batch = random.sample(input_set, batch_size)
        gradients_accumulator = [np.zeros(w.shape) for w in nnet.weights]
        error = 0.0

        for inputs, correct in batch:
            actual = nnet.value(inputs)
            error += nnet.prediction_error(correct, actual)
            gradients = [g for g in nnet.calculate_gradients(correct, actual)]

            error += (regularization_coeff / (2.0 * batch_size) *
                sum((g**2).sum() for g in gradients))

            for accumulator, gradient, weights in izip(gradients_accumulator,
                                                       gradients,
                                                       nnet.weights):
                accumulator += gradient + regularization_coeff * weights

        gradient = [np.clip(g / batch_size, -10, 10) for g in gradients_accumulator]
        error /= batch_size

        lrate = learning_rate(epoch)
        nnet.update_weights(gradient, lrate)

    return nnet
