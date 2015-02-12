from itertools import izip
import numpy as np
import random

def minibatch(nnet, input_set, batch_size, learning_rate, regularization_coeff, stop):
    if not hasattr(learning_rate, '__call__'):
        x = float(learning_rate)
        learning_rate = lambda _: x

    if not hasattr(stop, '__call__'):
        n = stop
        stop = lambda i, e: i > n

    epoch, error = 0, 1000.0
    while not stop(epoch, error):
        epoch += 1
        batch = random.sample(input_set, batch_size)
        gradients_accumulator = [ np.zeros(w.shape) for w in nnet.weights ]
        error = 0.0

        for inputs, correct in batch:
            actual = nnet.value(inputs)
            error += nnet.prediction_error(correct, actual)
            gradients = [g for g in nnet.calculate_gradients(correct, actual)]

            # regularization
            #error += (regularization_coeff / (2.0 * batch_size) *
            #    sum((g**2).sum() for g in gradients))

            for accumulator, gradient in izip(gradients_accumulator, gradients):
                #accumulator += (gradient * regularization_coeff) / batch_size
                accumulator += gradient / batch_size

        lrate = learning_rate(epoch)
        nnet.update_weights(gradients_accumulator, lrate)

    return nnet
