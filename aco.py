import numpy as np
import random

# Define the distance matrix
dist_matrix = np.array([[0, 2, 9, 10, 5],
                        [2, 0, 6, 4, 8],
                        [9, 6, 0, 7, 3],
                        [10, 4, 7, 0, 1],
                        [5, 8, 3, 1, 0]])

# Define the number of ants
num_ants = 10

# Define the pheromone matrix
pheromone_matrix = np.ones(dist_matrix.shape) / len(dist_matrix)

# Define the parameters of the algorithm
alpha = 1
beta = 5
evaporation_rate = 0.5
q = 100
num_iterations = 50


# Define the Ant System algorithm
def ant_system(dist_matrix, pheromone_matrix, alpha, beta, evaporation_rate, q, num_iterations):
    best_tour = None
    best_distance = float("inf")
    for i in range(num_iterations):
        tours = []
        tour_distances = []
        for j in range(num_ants):
            tour = construct_tour(dist_matrix, pheromone_matrix, alpha, beta)
            tour_distance = calculate_tour_distance(dist_matrix, tour)
            tours.append(tour)
            tour_distances.append(tour_distance)
            if tour_distance < best_distance:
                best_tour = tour
                best_distance = tour_distance
        update_pheromone_matrix(pheromone_matrix, tours, tour_distances, evaporation_rate, q)
    return best_tour, best_distance


# Define the tour construction algorithm
def construct_tour(dist_matrix, pheromone_matrix, alpha, beta):
    tour = []
    unvisited_cities = set(range(len(dist_matrix)))
    current_city = random.choice(list(unvisited_cities))
    unvisited_cities.remove(current_city)
    while unvisited_cities:
        next_city = choose_next_city(dist_matrix, pheromone_matrix, current_city, unvisited_cities, alpha, beta)
        tour.append((current_city, next_city))
        unvisited_cities.remove(next_city)
        current_city = next_city
    return tour


# Define the city selection algorithm
def choose_next_city(dist_matrix, pheromone_matrix, current_city, unvisited_cities, alpha, beta):
    city_probs = []
    total_prob = 0
    for city in unvisited_cities:
        pheromone = pheromone_matrix[current_city, city]
        distance = dist_matrix[current_city, city]
        prob = (pheromone ** alpha) * ((1 / distance) ** beta)
        city_probs.append((city, prob))
        total_prob += prob
    city_probs = [(city, prob / total_prob) for city, prob in city_probs]
    return max(city_probs, key=lambda x: x[1])[0]


# Define the pheromone update algorithm
def update_pheromone_matrix(pheromone_matrix, tours, tour_distances, evaporation_rate, q):
    pheromone_matrix *= evaporation_rate
    for tour, tour_distance in zip(tours, tour_distances):
        for city1, city2 in tour:
            pheromone_matrix[city1, city2] += q / tour_distance
            pheromone_matrix[city2, city1] = pheromone_matrix[city1, city2]


def calculate_tour_distance(dist_matrix, tour):
    distance = 0
    for i in range(len(tour)):
        city1, city2 = tour[i]
        distance += dist_matrix[city1, city2]
    return distance
