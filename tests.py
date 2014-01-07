from neuralnet import *

import matplotlib.pyplot as plt

def test_xor_value(nnet):
    # create or use the given neural network to compute the
    # xor function between its arguments. The network has
    # two inputs, one hidden layer with two neurons and 
    # one output

    sigmoid = lambda x: 1.0/(1.0+exp(-x))
    sigmoid_der = lambda x: sigmoid(x) * (1 - sigmoid(x))

    if nnet is None:
        nnet = NeuralNetwork([2, 2, 1], sigmoid, sigmoid_der)
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

    sigmoid = lambda x: 1.0/(1.0+exp(-x))
    sigmoid_der = lambda x: sigmoid(x) * (1 - sigmoid(x))

    nnet = NeuralNetwork([2, 2, 1], sigmoid, sigmoid_der)

    graph_data = []
    for i in range(1,10001):
        # a decreasing learning rate roughly halves the convergence time!
        learning_rate = 5000.0 / (5000.0 + i)

        sample = training_examples[randint(0, len(training_examples) -1)]
        err = nnet.backprop(sample[0], sample[1], learning_rate)

        if i % 100 == 0: graph_data.append((i, err, learning_rate))
        if i % 1000 == 0: print 'iteration ', i

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

    return nnet

if __name__ == '__main__':
    print 'creating a neural network and using it to compute the xor function:'
    test_xor_value(None)

    print '\ncreating a neural network and training it to compute the xor fn:'
    nnet = test_xor_learning()

    print '\ntesting the neural network just trained'
    test_xor_value(nnet)
