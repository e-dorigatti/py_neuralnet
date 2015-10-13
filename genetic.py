from neuralnet import NeuralNetwork
import numpy as np
import itertools


def combine(nnet1, nnet2, nnet_init_kwargs={}):
    new_weights = []
    for w1, w2 in itertools.izip(nnet1.weights, nnet2.weights):
        assert w1.shape == w2.shape
        t = np.random.random(w1.shape) * 3 - 1
        new_weights.append(t * w1 + (1 - t) * w2)

    nnet_size = [nnet1.weights[0].shape[1] - 1] + [w.shape[0] for w in nnet1.weights]
    new_nnet = NeuralNetwork(nnet_size, **nnet_init_kwargs)
    new_nnet.weights = new_weights

    assert all(nw.shape == w.shape 
               for nw, w in itertools.izip(new_nnet.weights, new_weights))

    return new_nnet


def genetic_learn(nnet_size, pop_size, fitness_fn, stop_fn, **nnet_kwargs):
    assert hasattr(fitness_fn, '__call__')
    if not hasattr(stop_fn, '__call__'):
        epoch_count = int(stop_fn)
        stop_fn = lambda i, p: i > n

    population = [NeuralNetwork(nnet_size, **nnet_kwargs) for _ in range(pop_size)]
    i, error, stop = 0, 0.0, False
    while not stop:
        new_pop = (combine(nnet1, nnet2, nnet_kwargs)
                   for nnet1, nnet2 in itertools.product(population, population))
        #ranked = sorted(itertools.chain(population, new_pop),
        ranked = sorted(new_pop,
                        key=lambda nnet: fitness_fn(nnet),
                        reverse=True)
        population = ranked[:pop_size]

        i += 1
        stop = stop_fn(i, population)

    return population[0]
