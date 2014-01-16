from math import exp

def sigmoid(x):
    return 1.0/(1.0+exp(-x))

def d_dx_sigmoid(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return (1 - exp(-2*x)) / (1 + exp(-2*x))

def d_dx_tanh(x):
    return 1 - tanh(x)**2

