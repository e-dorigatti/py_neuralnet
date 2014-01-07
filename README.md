py_neuralnet
============

Simple (&lt;100 loc) implementation of a feed forward neural network in python

Requirements: numpy and pyplot (only for the unit test)

The neural network is implemented as a linked list of layers, each containing
neurons as well as a bias unit. It is possible to specify the number of layers
and the number of neurons of each layer upon creation as well as an activation
function together with its first derivative (which only needed when learning).

The code is fully vectorized and takes advantage of numpy's arrays to perform
both forward-propagation and back-propagation, but information exchange with
the outside world is done through normal python lists.

A simple unit test is provided; it builds a simple neural network capable of
computing the exclusive-or between its arguments. The network has one hidden
layer composed of three neurons (including the bias term) and both forward
and back propagation are tested. The activation function used is the logistic
function, also known as sigmoid. The learning test plots a graph using pyplot.
