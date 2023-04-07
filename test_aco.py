import logging

import numpy as np

from aco import ACO


logging.basicConfig(level=logging.DEBUG)


def test_aco():
    # test
    distances = np.array([[0, 2, 9],
                          [2, 0, 6],
                          [9, 6, 0]])
    aco = ACO(n=2, iterations=2)
    # act
    tour = aco.solve(distances)
    # assert
    logging.info(tour)
    assert len(tour) == len(distances)
