import numpy as np
import random

def online_learn(nnet, train_set, epoch_length, learning_rate, stop_fn):
    """
    Tries to teach to the neural network the given training data using
    online learning.

    train_set: list of tuples (input vector, desired output vector) which
     constitute the training set for the neural network

    epoch_length: every epoch_length iterations stop the stop condition
     will be evaluated

    learning_rate: the learning rate to use for gradient descent. It can be
     a number or a function getting the current iteration number and returning
     the learning rate

    stop_fn: the stopping condition, evaluated every epoch_length iterations.
     It can be a number, in which case learning will be stopped after the
     specified iteration has been reached, or a function accepting the iteration,
     the average training error in the last epoch_length iteration and the
     current learning rate as parameters and returning True if the training
     has to be stopped.
    """
    if not hasattr(learning_rate, '__call__'):
        x = float(learning_rate)
        learning_rate = lambda i: x

    if not hasattr(stop_fn, '__call__'):
        n = stop_fn
        stop_fn = lambda i, e, l: i > n

    i, error, stop = 0, 0.0, False
    while not stop:
        inputs, correct = random.choice(train_set)
        outputs = nnet.value(inputs)

        lrate = learning_rate(i)
        error += nnet.backpropagate(correct, outputs, lrate) / epoch_length

        i += 1
        if i % epoch_length == 0:
            stop = stop_fn(i, error, lrate)
            error = 0.0
