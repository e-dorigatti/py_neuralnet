from math import exp
import numpy as np

def vectorize(f):
    return np.vectorize(f, otypes=[float])


class QuadraticError:
    @staticmethod
    def error(wanted_outputs, actual_outputs):
        return 0.5 * np.square(wanted_outputs - actual_outputs).sum()

    @staticmethod
    def deltas(wanted_outputs, actual_outputs):
        return wanted_outputs - actual_outputs


class CrossEntropyError:
    @staticmethod
    def error(wanted_outputs, actual_outputs):
        y, d = actual_outputs, wanted_outputs
        return -(d * np.log(y) + (1 - d) * np.log(1 - y)).sum()

    @staticmethod
    def deltas(wanted_outputs, actual_outputs):
        y, d = actual_outputs, wanted_outputs
        return (d / y) + (1 - d) / (1 - y)


# some common errors
common = {
    'quadratic': QuadraticError,
    'crossentropy': CrossEntropyError,
}
