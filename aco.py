import logging
import random

import numpy as np


logger = logging.getLogger('aco')
logging.basicConfig(level=logging.DEBUG)


class ACO:
    '''ant colony optimization specifically for the traveling salesman problem'''

    def __init__(self, n=10, alpha=1, beta=5, e=0.5, q=100, iterations=50):
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.e = e  # evaporation_rate
        self.q = q
        self.iterations = iterations

    def solve(self, distances):
        pheromones = np.ones(distances.shape) / len(distances)
        best_tour = None
        best_distance = float("inf")
        for i in range(self.iterations):
            tours = []
            tour_distances = []
            for j in range(self.n):
                tour = self.construct_tour(distances, pheromones)
                tour_distance = ACO.calculate_tour_distance(distances, tour)
                tours.append(tour)
                tour_distances.append(tour_distance)
                if tour_distance < best_distance:
                    best_tour = tour
                    best_distance = tour_distance
            self.update_pheromones(pheromones, tours, tour_distances)
        return best_tour, best_distance

    def construct_tour(self, distances, pheromones):
        tour = []
        unvisited_cities = set(range(len(distances)))
        current_city = random.choice(list(unvisited_cities))
        unvisited_cities.remove(current_city)
        while unvisited_cities:
            next_city = self.choose_next_city(distances, pheromones,
                                              current_city, unvisited_cities)
            tour.append((current_city, next_city))
            unvisited_cities.remove(next_city)
            current_city = next_city
        return tour

    def choose_next_city(self, distances, pheromones, current_city, unvisited_cities):
        city_probs = []
        total_prob = 0
        for city in unvisited_cities:
            pheromone = pheromones[current_city, city]
            distance = distances[current_city, city]
            prob = (pheromone ** self.alpha) * ((1 / distance) ** self.beta)
            city_probs.append((city, prob))
            total_prob += prob
        city_probs = [(city, prob / total_prob) for city, prob in city_probs]
        return max(city_probs, key=lambda x: x[1])[0]

    def update_pheromones(self, pheromones, tours, tour_distances):
        pheromones *= self.e
        for tour, tour_distance in zip(tours, tour_distances):
            for city1, city2 in tour:
                pheromones[city1, city2] += self.q / tour_distance
                pheromones[city2, city1] = pheromones[city1, city2]

    @staticmethod
    def calculate_tour_distance(distances, tour):
        distance = 0
        for i in range(len(tour)):
            city1, city2 = tour[i]
            distance += distances[city1, city2]
        return distance
