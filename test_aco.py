import logging

import numpy as np

from aco import ACO


logging.basicConfig(level=logging.DEBUG)


def test_aco():
    # test
    distances = np.random.randint(1, 10, size=(3, 3))
    distances = (distances + distances.T) / 2  # make it symmetrical
    aco = ACO()
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


def test_evaporation_rate():
    # arrange
    aco = ACO(e=0.5)
    pheromones = np.ones((3, 3))
    # act
    aco.update_pheromones(pheromones, [], [])
    # assert
    assert np.array_equal(pheromones, np.ones_like(pheromones)*0.5)


def test_next_city():
    # arrange equal pheromones
    aco = ACO()
    pheromones = np.ones((3, 3))
    distances = np.ones((3, 3))
    distances[0, 1] = 0.5
    expected = 1
    # act
    actual = aco.choose_next_city(distances, pheromones, 0, {1, 2})
    # assert
    assert expected == actual
    # arrange equal distances
    pheromones = np.ones((3, 3))
    distances = np.ones((3, 3))
    pheromones[0, 1] = 2
    expected = 1
    # act
    actual = aco.choose_next_city(distances, pheromones, 0, {1, 2})
    # assert
    assert actual == expected
