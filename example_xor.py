from neuralnet import *
import errors, activations

from random import randint, uniform
import matplotlib.pyplot as plt


def setup_plot():
    fig, ax = plt.subplots()
    axis_error, axis_lrate = [ax, ax.twinx()]

    fig.subplots_adjust(right=0.85); axis_error.set_xlabel('Iterations')
    axis_error.set_ylabel('Training error'); axis_error.tick_params(axis='y')

    axis_lrate.set_ylabel('Learning rate'); axis_lrate.tick_params(axis='y')
    axis_lrate.set_ylim(0, 2)

    plt.title('Training error and learning rate vs. iterations')

    return axis_error, axis_lrate


def main():
    """
    Create a simple network with one hidden layer and train it to
    compute the xor between its arguments.
    """

    training_examples = [
        ([0, 0], [0]),
        ([0, 1], [1]),
        ([1, 0], [1]), 
        ([1, 1], [0])]

    nnet = NeuralNetwork([2, 2, 1])

    history, i, avg_error = list(), 1, 1
    axis_error, axis_lrate = setup_plot()
    errors, lrates = list(), list()

    while avg_error > 0.0025 and i < 25000:
        i += 1
        learning_rate = 10000.0 / (5000.0 + i)

        input, correct_output = training_examples[randint(0, len(training_examples) - 1)]
        actual_output = nnet.value(input)
        history.append(nnet.backpropagate(correct_output, actual_output, learning_rate))

        if i % 1000 == 0: print 'iteration ', i, 'error', avg_error
        if i % 100 == 0:
            avg_error = sum(history) / float(len(history))
            history = []

            errors.append(avg_error)
            lrates.append(learning_rate)


    axis_lrate.plot(range(0, i, 100), lrates, '-g', label='Learning rate')
    axis_error.plot(range(0, i, 100), errors, '-b', label='Training error')

    h0, l0 = axis_error.get_legend_handles_labels()
    h1, l1 = axis_lrate.get_legend_handles_labels()
    plt.legend([h0[0], h1[0]], [l0[0], l1[0]])

    plt.show()


if __name__ == '__main__':
    main()
