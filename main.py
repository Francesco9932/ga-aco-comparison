from ga_tsp import Genetic
from ant_tsp import AntColony
import time
import numpy as np
from utils import generate_cities, make_mat, change_mat_diag_elem
from visualize import plot_statistics

nTest = 30
nCities = 50


def main():
    elapsed_times_GA = []
    elapsed_times_ACO = []
    best_dists_GA = []
    best_dists_ACO = []

    for i in range(1, nTest+1):
        # Inizializzazione random della matrice delle distanze tra
        # una città x e una città y
        cities = range(1, nCities+1)
        city_coordinates = generate_cities(len(cities))
        adjacency_mat = make_mat(city_coordinates)
        adjacency_mat = change_mat_diag_elem(adjacency_mat, np.inf)

        tsp_ga = Genetic(distances=adjacency_mat,
                         pop_size=140, max_generation=136)
        tsp_aco = AntColony(adjacency_mat, 20, 1, 50,
                            0.9, alpha=1.2, beta=1.2)

        # GA-TSP
        print("GA {} test: ".format(i))
        st = time.time()
        best_dists_GA.append(tsp_ga.geneticAlgorithm().fitness)
        et = time.time()
        elapsed_times_GA.append(et-st)
        print("\n")

        # ACO-TSP
        print("ACO {} test: ".format(i))
        st = time.time()
        best_dists_ACO.append(tsp_aco.run()[1])
        et = time.time()
        elapsed_times_ACO.append(et-st)
        print("\n")

    plot_statistics(nTest, elapsed_times_GA, elapsed_times_ACO,
                    best_dists_GA, best_dists_ACO)


if __name__ == "__main__":
    main()
