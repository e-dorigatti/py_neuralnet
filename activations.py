from math import exp
import numpy as np

def vectorize(f):
    return np.vectorize(f, otypes=[float])


class SigmoidActivation:
    @staticmethod
    @vectorize
    def f(x):
        return 1.0 / (1.0 + exp(-x))

    @staticmethod
    @vectorize
    def f_prime(x):
        return SigmoidActivation.f(x) * (1 - SigmoidActivation.f(x))


class TanhActivation:
    @staticmethod
    @vectorize
    def f(x):
        return (1 - exp(-2*x)) / (1 + exp(-2*x))

    @staticmethod
    @vectorize
    def f_prime(x):
        return 1 - TanhActivation.f(x)**2


# some common activations
common = {
    'sigmoid': SigmoidActivation,
    'tanh': TanhActivation,
}
