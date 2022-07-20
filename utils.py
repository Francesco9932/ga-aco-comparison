import numpy as np


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


def change_mat_diag_elem(dist_matrix, change):
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            if i == j:
                dist_matrix[i][j] = change
            else:
                dist_matrix[i][j] = int(dist_matrix[i][j])
    return dist_matrix
