import numpy as np
from numpy.random import choice as np_choice


class AntColony(object):

    def __init__(self, distances, n_ants, n_best, n_iterations, decay, alpha=1, beta=1):
        """
        Args:
            distances (2D numpy.array): Matrice quadrata delle distanze. Si assume np.inf nella diagonale.
            n_ants (int): Numero di agenti per ogni iterazione
            n_best (int): Numero di agenti con migliori performance che possono rilasciare feromone
            n_iteration (int): Numero iterazioni
            decay (float): Pheromone decay (es. 0.95). Il valore del feromone viene moltiplicato per il decay.
            alpha (int or float): valore dell'esponente per il feronome (default = 1)
            beta (int or float): valore dell'esponente per la distanza (default = 1)

        Esempio:
            ant_colony = AntColony(german_distances, 100,
                                   20, 2000, 0.95, alpha=1, beta=2)
        """
        self.distances = distances
        self.pheromone = np.ones(self.distances.shape) / len(distances)
        self.all_inds = range(len(distances))
        self.n_ants = n_ants
        self.n_best = n_best
        self.n_iterations = n_iterations
        self.decay = decay
        self.alpha = alpha
        self.beta = beta

    # funzione principale richiamata dall'esterno della classe una volta
    # inizializzata la classe
    def run(self):
        shortest_path = None
        all_time_shortest_path = ("placeholder", np.inf)

        for i in range(self.n_iterations):
            all_paths = self.gen_all_paths()
            self.spread_pheronome(all_paths, self.n_best,
                                  shortest_path=shortest_path)
            shortest_path = min(all_paths, key=lambda x: x[1])

            if i % 5 == 0:
                print("dist: {}".format(shortest_path[1]))

            if shortest_path[1] < all_time_shortest_path[1]:
                all_time_shortest_path = shortest_path
            self.pheromone = self.pheromone * self.decay

        return all_time_shortest_path

    # aggiornamento del feromone
    def spread_pheronome(self, all_paths, n_best, shortest_path):
        sorted_paths = sorted(all_paths, key=lambda x: x[1])
        for path, dist in sorted_paths[:n_best]:
            for move in path:
                self.pheromone[move] += 1.0 / self.distances[move]

    # calcolo della distanza
    def gen_path_dist(self, path):
        total_dist = 0
        for ele in path:
            total_dist += self.distances[ele]
        return total_dist

    def gen_all_paths(self):
        all_paths = []
        for i in range(self.n_ants):
            path = self.gen_path(0)
            all_paths.append((path, self.gen_path_dist(path)))
        return all_paths

    def gen_path(self, start):
        path = []
        visited = set()
        visited.add(start)
        prev = start
        for i in range(len(self.distances) - 1):
            move = self.pick_move(
                self.pheromone[prev], self.distances[prev], visited)
            path.append((prev, move))
            prev = move
            visited.add(move)
        path.append((prev, start))  # torna da dove siamo piartiti
        return path

    # determina la mossa per ogni agente
    def pick_move(self, pheromone, dist, visited):
        pheromone = np.copy(pheromone)
        pheromone[list(visited)] = 0

        row = pheromone ** self.alpha * ((1.0 / dist) ** self.beta)

        norm_row = row / row.sum()
        move = np_choice(self.all_inds, 1, p=norm_row)[0]
        return move


''' distances = np.array([[np.inf, 2, 2, 5, 7],
                      [2, np.inf, 4, 8, 2],
                      [2, 4, np.inf, 1, 3],
                      [5, 8, 1, np.inf, 2],
                      [7, 2, 3, 2, np.inf]])

cityCoordinates = [[5, 80], [124, 31], [46, 54], [86, 148], [21, 8],
                   [134, 72], [49, 126], [36, 34], [26, 49], [141, 6],
                   [124, 122], [80, 92], [70, 69], [76, 133], [23, 65]]

distance_matrix = spatial.distance.cdist(
    cityCoordinates, cityCoordinates, metric='euclidean')


def dist_matrix_infinity_diagonal_elements(dist_matrix):
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            if i == j:
                dist_matrix[i][j] = np.inf
            else:
                dist_matrix[i][j] = int(dist_matrix[i][j])
    return dist_matrix '''


''' distance_matrix = np.asarray(
    [
        [np.inf, 28.02, 17.12, 27.46, 46.07],
        [28.02, np.inf, 34.00, 25.55, 25.55],
        [17.12, 34.00, np.inf, 18.03, 57.38],
        [27.46, 25.55, 18.03, np.inf, 51.11],
        [46.07, 25.55, 57.38, 51.11, np.inf],
    ])

ant_colony = AntColony(distance_matrix, 1, 1, 100, 0.95, alpha=1, beta=1)
shortest_path = ant_colony.run()
print("shorted_path: {}".format(shortest_path)) '''
