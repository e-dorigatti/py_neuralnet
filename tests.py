from neuralnet import *
from utils import *

from random import randint, uniform
import matplotlib.pyplot as plt

def test_xor_value(nnet):
    # create or use the given neural network to compute the
    # xor function between its arguments. The network has
    # two inputs, one hidden layer with two neurons and 
    # one output

    if nnet is None:
        nnet = NeuralNetwork([2, 2, 1], sigmoid, d_dx_sigmoid)
        nnet.layers[1].weights = np.array([ \
            [ 30, -20, -20 ],
            [-10,  20,  20 ]])
        nnet.layers[2].weights = np.array([ \
            [ -30, 20, 20 ]])

    print '0 xor 0 = ', nnet.value([0, 0])[0]    # circa 0
    print '0 xor 1 = ', nnet.value([0, 1])[0]    # circa 1
    print '1 xor 0 = ', nnet.value([1, 0])[0]    # circa 1
    print '1 xor 1 = ', nnet.value([1, 1])[0]    # circa 0

def test_xor_learning():
    # create a simple network with one hidden layer and train
    # it to compute the xor between its arguments

    training_examples = [ \
        [[0, 0], [0]], \
        [[0, 1], [1]], \
        [[1, 0], [1]], \
        [[1, 1], [0]]]

    nnet = NeuralNetwork([2, 2, 1], activation = 'sigmoid')

    graph_data = []
    last = []
    i = avg_error = 1

    while avg_error > 0.0025 and i < 50000:
        i += 1
        learning_rate = 5000.0 / (5000.0 + i)

        sample = training_examples[randint(0, len(training_examples) -1)]
        last.append(nnet.backprop(sample[0], sample[1], learning_rate))

        if i % 1000 == 0: print 'iteration ', i
        if i % 100 == 0:
            avg_error = sum(last) / float(len(last))
            last = []
            graph_data.append((i, avg_error, learning_rate))

    # plot a graph with error and learning rate
    fig, ax = plt.subplots()
    axes = [ax, ax.twinx()]
    fig.subplots_adjust(right=0.85)
    axes[0].set_xlabel('Iterations')

    # plot error on its own axis
    axes[0].plot([i for i,e,l in graph_data], [e for i,e,l in graph_data], \
        '-b', label = 'Training error')
    axes[0].set_ylabel('Training error')
    axes[0].tick_params(axis = 'y')

    # plot learning rate on its own axis
    axes[1].plot([i for i,e,l in graph_data], [l for i,e,l in graph_data], \
        '-g', label = 'Learning rate')
    axes[1].set_ylabel('Learning rate')
    axes[1].tick_params(axis = 'y')
    axes[1].set_ylim(0, 1)

    # create a legend
    h0, l0 = axes[0].get_legend_handles_labels()
    h1, l1 = axes[1].get_legend_handles_labels()
    plt.legend([h0[0], h1[0]], [l0[0], l1[0]])

    plt.title('Training error and learning rate vs. iterations')
    plt.show()

    return avg_error, nnet

def test_derivatives():
    function = lambda x: (x**2)
    nnet = NeuralNetwork([1, 2, 1])

    i = avg_error = 1
    last = []
    while i < 25000:
        i += 1
        learning_rate = 5000.0 / (5000.0 + i)

        input = [ uniform(-1, 1) ]
        output = [ function(input[0]) ]
        last.append(nnet.backprop(input, output, learning_rate))

        if i % 1000 == 0:
            avg_error = sum(last) / len(last)
            last = []
            print 'iteration', i, 'error', avg_error, 'learning rate', learning_rate

    plot = { 
        'x': [], 'f': [],
        'nnet': [], 'nnet_dx': [],
        'dq': [] # difference quotient for estimating the derivative
    }

    h = 0.01
    for x in np.arange(-1, 1, h):
        plot['x'].append(x)
        plot['f'].append(function(x))
        plot['nnet'].append(nnet.value([x])[0])
        plot['nnet_dx'].append(nnet.output_derivatives()[0][0])
        if len(plot['x']) > 1:
            plot['dq'].append( (plot['nnet'][-1] - plot['nnet'][-2]) / h)
    plot['dq'].append( (plot['nnet'][-1] - plot['nnet'][-2]) / h)

    plt.figure()
    plt.subplot(1, 2, 1)
    plt.plot(plot['x'], plot['f'], label='function')
    plt.plot(plot['x'], plot['nnet'], label='learned function')
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(plot['x'], plot['nnet_dx'], c='b', ls='-', label='derivative from nnetwork')
    plt.plot(plot['x'], plot['dq'], c='r', ls='--', label='analytical derivative')
    plt.legend()
    plt.grid()

    plt.show()

if __name__ == '__main__':
    print 'creating a neural network and using it to compute the xor function:'
    test_xor_value(None)

    print '\ncreating a neural network and training it to compute the xor fn:'
    error, nnet = test_xor_learning()
    print 'neural network successfully trained with error', error

    print '\ntesting the neural network just trained'
    test_xor_value(nnet)

    print '\ntraining a neural network with the function x**2 and testing the derivatives'
    test_derivatives()

