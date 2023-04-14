import logging

import numpy as np

from aco import ACO


logging.basicConfig(level=logging.DEBUG)


def test_aco():
    # test
    distances = np.array([[0, 2.23606798, 3.16227766, 3.60555128],
                          [2.23606798, 0, 2.23606798, 2.82842712],
                          [3.16227766, 2.23606798, 0, 2.23606798],
                          [3.60555128, 2.82842712, 2.23606798, 0]])
    aco = ACO(n=2, iterations=2)
    upper_bound = np.sum(np.amax(distances, axis=1))
    # act
    tour, cost = aco.solve(distances)
    # assert
    assert cost <= upper_bound
    assert len(tour) == len(distances)


def test_construct_tour():
    # arrange
    distances = np.array([[0, 2, 9],
                          [2, 0, 6],
                          [9, 6, 0]])
    aco = ACO()
    pheromones = np.ones(distances.shape) / len(distances)
    # act
    tour = aco.construct_tour(distances, pheromones)
    # assert
    assert len(tour) == len(distances)


def test_update_pheromoes():
    # arrange
    aco = ACO()
    # assert
    assert 0


def test_next_city():
    assert 0
