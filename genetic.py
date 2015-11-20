from neuralnet import NeuralNetwork
import numpy as np
import itertools


class GeneticAlgorithm:
    def fitness(self, nnet):
        """ Performance evaluation of a single neural network; the higher the better
        """
        raise NotImplemented()

    def stop(self, epoch, population):
        """ Stopping criterion for learning.
        Population is sorted by fitness in descending order.
        """
        raise NotImplemented()

    def get_new_neural_network(self, size):
        return NeuralNetwork(size)

    def initial_population(self, nnet_size, pop_size):
        return [self.get_new_neural_network(nnet_size) for _ in range(pop_size)]

    def combine(self, nnet1, nnet2):
        assert nnet1.ns == nnet2.ns

        new_weights = []
        size = [nnet1.weights[0].shape[1] - 1]
        for w1, w2 in itertools.izip(nnet1.weights, nnet2.weights):
            assert w1.shape == w2.shape
            t = np.random.random(w1.shape) * 3 - 1
            new_weights.append(t * w1 + (1 - t) * w2)

        new_nnet = self.get_new_neural_network(nnet1.ns)
        new_nnet.weights = new_weights
        return new_nnet


def genetic_learn(algo, nnet_size, pop_size):
    population = algo.initial_population(nnet_size, pop_size)
    i, stop = 0, False
    while not stop:
        new_pop = (algo.combine(nnet1, nnet2)
                   for nnet1, nnet2 in itertools.product(population, population))
        ranked = sorted(new_pop, key=algo.fitness, reverse=True)
        population = ranked[:pop_size]

        i += 1
        stop = algo.stop(i, population)

    return population[0]


def genetic_learn_spark(sc, algo, nnet_size, pop_size, num_slices=4):
    """ Evolutionary learning for neural networks implemented with spark.
    First argument is the spark context.
    """
    population = algo.initial_population(nnet_size, pop_size)
    i, stop = 0, False
    while not stop:
        netrdd = sc.parallelize(population, num_slices)
        temp = (netrdd.cartesian(netrdd)
            .map(lambda (nnet1, nnet2): algo.combine(nnet1, nnet2))
            .keyBy(algo.fitness)
            .persist())  # this somehow avoids triple sorting
        population = (temp.sortByKey(ascending=False)
            .values()
            .take(pop_size))
        temp.unpersist()

        i += 1
        stop = algo.stop(i, population)

    return population[0]
