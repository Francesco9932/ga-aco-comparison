import numpy as np
import random


class Genome():
    def __init__(self):
        self.chromosomes = []
        self.fitness = np.inf


class Genetic(object):
    def __init__(self, distances, pop_size=100, elite_size=10, max_generation=100,
                 mutation_rate=0.03):
        self.distances = distances
        self.pop_size = pop_size
        self.max_generation = max_generation
        self.mutation_rate = mutation_rate
        self.elite_size = elite_size

    def createNewPopulation(self, size):
        population = []
        for x in range(size):
            newGenome = Genome()
            # Inzializza random gli indici delle città per ogni cromosoma
            newGenome.chromosomes = random.sample(
                range(1, len(self.distances)), len(self.distances)-1)
            # Aggiunge uno zero all'inizio
            newGenome.chromosomes.insert(0, 0)
            # Aggiunge uno zero alla fine
            newGenome.chromosomes.append(0)
            # Calcola il funzionale del cromosoma
            newGenome.fitness = self.fitness(newGenome.chromosomes)
            # Aggiunge il cromosoma alla popolaziones
            population.append(newGenome)
        return population

    # Calcola la fitness: totale delle distanze

    def fitness(self, chromosomes):
        return sum(
            [self.distances[chromosomes[i], chromosomes[i + 1]]
                for i in range(len(chromosomes) - 1)]
        )

    # Restituisce il cromosoma con il funzionale più piccolo

    def findBestGenome(self, population):
        allFitness = [i.fitness for i in population]
        bestFitness = min(allFitness)
        return population[allFitness.index(bestFitness)]

    # In K-Way tournament selection, we select K individuals
    # from the population at random and select the best out
    # of these to become a parent. The same process is repeated
    # for selecting the next parent.

    def tournamentSelection(self, population, k=4):
        # Seleziona random k individui
        selected = [population[random.randrange(
            0, len(population))] for i in range(k)]
        # Prende il migliore tra i k individui selezionati
        bestGenome = self.findBestGenome(selected)
        return bestGenome

    # Restituisce un cromosoma figlio

    def reproduction(self, population):
        parent1 = self.tournamentSelection(population, 10).chromosomes
        parent2 = self.tournamentSelection(population, 6).chromosomes
        while parent1 == parent2:
            parent2 = self.tournamentSelection(population, 6).chromosomes

        return self.orderOneCrossover(parent1, parent2)

    # Effettua un incrocio e una mutazione
    # Sample:
    # parent1 = [0, 3, 8, 5, 1, 7, 12, 6, 4, 10, 11, 9, 2, 0]
    # parent2 = [0, 1, 6, 3, 5, 4, 10, 2, 7, 12, 11, 8, 9, 0]
    # child   = [0, 1, 3, 5, 2, 7, 12, 6, 4, 10, 11, 8, 9, 0]

    def orderOneCrossover(self, parent1, parent2):
        size = len(parent1)
        child = [-1] * size

        child[0], child[size - 1] = 0, 0

        idx = np.random.choice(range(size-1), size=2, replace=False)
        start, end = min(idx), max(idx)

        for i in range(start, end + 1):
            child[i] = parent1[i]
        point = end+1
        point2 = start
        while child[point] in [-1, 0]:
            if child[point] != 0:
                if parent2[point2] not in child:
                    child[point] = parent2[point2]
                    point += 1
                    if point == size:
                        point = 0
                else:
                    point2 += 1
                    if point2 == size:
                        point2 = 0
            else:
                point += 1
                if point == size:
                    point = 0

        # Effettua una mutazione del figlio ottenuto, con un tasso
        # di mutazione dato da MUTATION_RATE
        if random.random() <= self.mutation_rate:
            child = self.swapMutation(child)

        # Create new genome for child
        newGenome = Genome()
        newGenome.chromosomes = child
        newGenome.fitness = self.fitness(child)
        return newGenome

    # Sample:
    # Chromosomes =         [0, 3, 8, 5, 1, 7, 12, 6, 4, 10, 11, 9, 2, 0]
    # Mutated chromosomes = [0, 11, 8, 5, 1, 7, 12, 6, 4, 10, 3, 9, 2, 0]

    def swapMutation(self, chromo):
        # for x in range(self.mutation_repeat_count):
        p1, p2 = [random.randrange(1, len(chromo) - 1) for i in range(2)]
        while p1 == p2:
            p2 = random.randrange(1, len(chromo) - 1)
        temp = chromo[p1]
        chromo[p1] = chromo[p2]
        chromo[p2] = temp
        return chromo

    def geneticAlgorithm(self):
        #allBestFitness = []
        population = self.createNewPopulation(self.pop_size)
        generation = 0
        # Ciclo principale
        while generation < self.max_generation:
            generation += 1

            offsprings = []
            elite_population = sorted(population, key=lambda i: i.fitness)[
                :self.elite_size]

           # Incrocio
            for i in range(self.elite_size, self.pop_size):
                # Select parent, make crossover and
                # after, append in population a new child
                offsprings.append(self.reproduction(population))

            population = offsprings + elite_population
            random.shuffle(population)

            # Fitness media della popolazione
            ''' averageFitness = round(
                np.sum([genom.fitness for genom in population]) / len(population), 2) '''
            # Prende il miglior cromosoma
            bestGenome = self.findBestGenome(population)
            print("\n" * 5)
            print("Generation: {0}\nPopulation Size: {1}\nBest Fitness(distance): {2}"
                  .format(generation, len(population),  # averageFitness,
                          bestGenome.fitness))

            # Tiene traccia di tutte le migliori fitness di ogni generazione per poi
            # tracciare la curva
            # allBestFitness.append(bestGenome.fitness)

        return bestGenome
        # Visualize
        # plot(generation, allBestFitness, bestGenome, cityCoordinates)
