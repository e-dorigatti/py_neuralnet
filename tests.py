from neuralnet import *
import errors, activations
from genetic import genetic_learn
from minibatch import minibatch
from random import randint, uniform
import matplotlib.pyplot as plt


test_list = list()
def testmethod(function):
    def wrapped(*args, **kwargs):
        print(function.__name__.replace('_', ' '))
        try:
            function(*args, **kwargs)
            print('PASS\n')
        except AssertionError as ae:
            print('FAIL (%s)\n' % (ae.message or '<no message>'))
    
    test_list.append(wrapped)
    return function


@testmethod
def test_forward_propagation(*args, **kwargs):
    nnet = NeuralNetwork([2, 2, 1])
    nnet.weights = [ np.array([
            [ 30, -20, -20 ],
            [-10,  20,  20 ]]),
        np.array([
            [ -30, 20, 20 ]]) ]

    assert nnet.value([0, 0])[0][0] < 0.05
    assert nnet.value([1, 0])[0][0] > 0.95
    assert nnet.value([0, 1])[0][0] > 0.95
    assert nnet.value([1, 1])[0][0] < 0.05


@testmethod
def test_backpropagation(training_examples):
    nnet = NeuralNetwork([2, 2, 1])

    i, avg_error, last = 0, 100.0, list()
    while avg_error > 0.0025 and i < 25000:
        i += 1
        learning_rate = 10

        input, correct_output = training_examples[randint(0, len(training_examples) -1)]
        actual_output = nnet.value(input)
        last.append(nnet.backpropagate(correct_output, actual_output, learning_rate))

        if i % 100 == 0:
            avg_error = sum(last) / float(len(last))
            last = []

    if i == 25000:
        print('iteration limit reached, learning might not have converged')

    for input, correct in training_examples:
        val = nnet.value(input)[0][0]
        assert abs(val - correct[0]) < 0.1


@testmethod
def test_derivatives(training_examples):
    nnet = NeuralNetwork([2, 4, 2])

    def get_derivatives(inputs):
        _ = nnet.value(inputs)
        return nnet.output_derivatives()

    epsilon = 0.0001
    for x in np.arange(-1, 1, 0.1):
        for y in np.arange(-1, 1, 0.1):
            nnet_deriv = get_derivatives([x, y])

            x_deriv = (get_derivatives([x + epsilon, y]) -
                get_derivatives([x - epsilon, y])) / (2 * epsilon)
            y_deriv = (get_derivatives([x, y + epsilon]) - 
                get_derivatives([x, y - epsilon])) / (2 * epsilon)

            assert ((nnet_deriv - x_deriv)**2).sum() < 0.05
            assert ((nnet_deriv - y_deriv)**2).sum() < 0.05


@testmethod
def test_minibatch(training_examples):
    nnet = NeuralNetwork([2, 2, 1])

    def stop(epoch, error, *args, **kwargs):
        print(epoch, error)
        return epoch > 50 or error < 0.01

    def learning_rate(epoch):
        return  3000.0 / (1000 + epoch)

    minibatch(nnet, training_examples, len(training_examples), 2, 0, stop)

    for input, correct in training_examples:
        val = nnet.value(input)[0][0]
        assert abs(val - correct[0]) < 0.1


@testmethod
def test_genetic(training_examples):
    nnet_size = [2, 2, 1]

    def fitness(nnet):
        error = sum(((nnet.value(inputs) - correct)**2).sum()
                    for inputs, correct in training_examples)
        return -error

    def stop(i, pop):
        return i > 50 or fitness(pop[0]) > -0.001

    nnet = genetic_learn(nnet_size, 25, fitness, stop)

    for input, correct in training_examples:
        val = nnet.value(input)[0][0]
        assert abs(val - correct[0]) < 0.1


if __name__ == '__main__':
    training_examples = [([0, 0], [0]), ([0, 1], [1]), ([1, 0], [1]), ([1, 1], [0])]
    for test in test_list:
        test(training_examples)
