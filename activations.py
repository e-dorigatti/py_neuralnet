from math import exp
import numpy as np

def vectorize(f):
    return np.vectorize(f, otypes=[float])


class SigmoidActivation:
    @staticmethod
    @vectorize
    def f(x):
        try:
            return 1.0 / (1.0 + exp(-x))
        except OverflowError:
            return 1 if x < 0 else 0

    @staticmethod
    @vectorize
    def f_prime(x):
        return SigmoidActivation.f(x) * (1 - SigmoidActivation.f(x))


class TanhActivation:
    @staticmethod
    @vectorize
    def f(x):
        try:
            return (1 - exp(-2*x)) / (1 + exp(-2*x))
        except OverflowError:
            return 1 if x > 0 else -1

    @staticmethod
    @vectorize
    def f_prime(x):
        return 1 - TanhActivation.f(x)**2


# some common activations
common = {
    'sigmoid': SigmoidActivation,
    'tanh': TanhActivation,
}
