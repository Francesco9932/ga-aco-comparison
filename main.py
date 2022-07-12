from ga_tsp import Genetic
from ant_tsp import AntColony
import numpy as np
from statistics import mean
import time

nTest = 30
nCities = 75


def generate_cities(n_cities, factor=10):
    return np.random.rand(n_cities, 2) * n_cities * factor


def make_mat(coordinates):
    res = [
        [get_distance(city1, city2) for city2 in coordinates]
        for city1 in coordinates
    ]
    return np.asarray(res)


def get_distance(city1, city2):
    return np.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)


def dist_matrix_infinity_diagonal_elements(dist_matrix):
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            if i == j:
                dist_matrix[i][j] = np.inf
            else:
                dist_matrix[i][j] = int(dist_matrix[i][j])
    return dist_matrix


def main():
    list_elapsed_time_GA = []
    list_elapsed_time_ACO = []
    list_best_dist_GA = []
    list_best_dist_ACO = []

    for i in range(nTest+1):
        # Inizializzazione random della matrice delle distanze tra
        # una città x e una città y
        cities = range(nCities)
        city_coordinates = generate_cities(len(cities))
        adjacency_mat = make_mat(city_coordinates)
        adjacency_mat = dist_matrix_infinity_diagonal_elements(adjacency_mat)

        tsp_ga = Genetic(distances=adjacency_mat,
                         pop_size=140, max_generation=136)
        tsp_aco = AntColony(adjacency_mat, 20, 1, 50,
                            0.9, alpha=1.2, beta=1.2)

        # GA-TSP
        print("GA {} test: ".format(i))
        time.sleep(1)
        st = time.time()
        ga_shortest_path = tsp_ga.geneticAlgorithm()
        et = time.time()
        elapsed_time_ga = et-st
        list_elapsed_time_GA.append(elapsed_time_ga)
        list_best_dist_GA.append(ga_shortest_path.fitness)
        time.sleep(1)
        print("\n")

        # ACO-TSP
        print("ACO {} test: ".format(i))
        st = time.time()
        aco_shortest_path = tsp_aco.run()
        et = time.time()
        elapsed_time_aco = et-st
        list_elapsed_time_ACO.append(elapsed_time_aco)
        list_best_dist_ACO.append(aco_shortest_path[1])
        time.sleep(1)
        print("\n")

    # Calcolo del tempo medio di esecuzione di GA e ACO
    avg_elapsed_time_GA = mean(list_elapsed_time_GA)
    avg_elapsed_time_ACO = mean(list_elapsed_time_ACO)

    # Calcolo della media della migliore distanza trovata da GA e ACO
    avg_best_dist_GA = mean(list_best_dist_GA)
    avg_best_dist_ACO = mean(list_best_dist_ACO)

    print("GA avg time: {}s. ACO avg time: {}s.".format(
        round(avg_elapsed_time_GA, 2), round(avg_elapsed_time_ACO, 2)))
    print("GA avg best distance: {}. ACO avg best distance: {}.".format(
        round(avg_best_dist_GA, 2), round(avg_best_dist_ACO, 2)))


if __name__ == "__main__":
    main()
