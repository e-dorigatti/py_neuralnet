from itertools import izip
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
        gradients_accumulator = None
        error = 0.0

        for input, correct in batch:
            actual = nnet.value(input)
            error += nnet.error.error(correct, actual) / batch_size
            gradients = [ g for g in nnet.calculate_gradients(correct, actual)]

            # regularization
            error += (regularization_coeff / (2 * batch_size) *
                sum([(g**2).sum() for g in gradients]))
            for weights, gradient in izip(nnet.weights, gradients):
                gradient += gradient * regularization_coeff

            if gradients_accumulator is not None:
                for accumulator, gradient in izip(gradients_accumulator, gradients):
                    accumulator += gradient / batch_size
            else:
                gradients_accumulator = gradients

        lrate = learning_rate(epoch)
        nnet.update_weights(gradients_accumulator, lrate)

    return nnet
