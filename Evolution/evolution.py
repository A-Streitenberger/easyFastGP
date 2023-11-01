import time

import numpy as np
from numba import njit

from Base.config.parameters import Parameters
from Base.individual import Individual
from Base.program import get_height, function_set
from Evolution.interfaces.crossover import CrossoverInterface
from Evolution.interfaces.fitness import FitnessInterface
from Evolution.interfaces.mutation import MutationInterface
from Evolution.interfaces.selection import SelectionInterface

"""
The class evolves a initial population over several generations.
It saves intermediate results.
"""


class Evolution:
    __slots__ = ["config", "crossover", "mutation", "selection", "fitness", "_generations", "_start_time"]

    def __init__(self, config: Parameters, crossover: CrossoverInterface, mutation: MutationInterface,
                 selection: SelectionInterface, fitness: FitnessInterface):
        self.config = config
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.fitness = fitness
        self._generations = None
        self._start_time = None

    def execute(self, init_population: list[Individual]) -> Individual:
        self._start_time = time.time()
        self._generations = 0

        self.__simple_logging(init_population)
        queue_population = init_population

        while not self.__should_terminate():
            elites = self.__get_elites(queue_population)

            new_pop = []
            new_pop.extend(elites)

            mothers = self.selection.execute(queue_population)

            for i in range(self.config.pop_size - self.config.elitism):
                # input for offspring: mother + random father
                new_pop.append(self.__generate_offspring(mothers[i], queue_population[
                    randint(self.config.pop_size)]))

            fitness_values = self.fitness.evaluate_population(new_pop)
            for individual, fitness_value in zip(new_pop, fitness_values):
                individual.fitness = fitness_value

            queue_population = new_pop
            self.__simple_logging(queue_population)
            self._generations += 1

        return min(queue_population, key=lambda indiv: indiv.fitness)

    def __get_elites(self, queue_population):
        elites = []
        if self.config.elitism > 0:
            elites = sorted(queue_population, key=lambda indiv: indiv.fitness)[:self.config.elitism]
        return elites

    def __should_terminate(self):
        if 0 < self.config.max_generations <= self._generations:
            return True
        if 0 < self.config.max_time <= time.time() - self._start_time:
            return True

        return False

    def __generate_offspring(self, mother: Individual, father: Individual) -> Individual:
        child = None
        if np.random.rand() < self.config.crossover_rate:
            child = self.crossover.execute(mother, father.program)

        if np.random.rand() < self.config.mutation_rate:
            if not child:
                child = mother

            child = self.mutation.execute(child)

        return child if child and self.__check_run_configuration(child) else mother

    def __check_run_configuration(self, individual: Individual):
        length = len(individual.program.nodes)

        if 0 < self.config.max_size < length:
            return False

        height = get_height(individual.program.nodes, function_set)
        individual.height = height

        if 0 < self.config.max_height < height:
            return False

        if height < self.config.min_height:
            return False

        return True

    def __simple_logging(self, population):
        heights = np.array(
            [individual.height if individual.height is not None else get_height(individual.program.nodes, function_set)
             for individual in population])
        nodes = np.array([len(individual.program.nodes) for individual in population])
        fitness_values = np.array([individual.fitness for individual in population])

        min_height = np.min(heights)
        max_height = np.max(heights)
        avg_height = np.mean(heights)

        min_nodes = np.min(nodes)
        max_nodes = np.max(nodes)
        avg_nodes = np.mean(nodes)

        min_fitness = np.min(fitness_values)
        max_fitness = np.max(fitness_values)
        avg_fitness = np.mean(fitness_values)

        with open('Logging_GP_run', 'a') as file:
            file.write(
                f"{self._generations};{min_nodes};{max_nodes};{avg_nodes:.20f};{min_height};{max_height};"
                f"{avg_height:.20f};{min_fitness:.20f};{max_fitness:.20f};{avg_fitness:.20f}\n")


@njit(cache=True)
def randint(size):
    return np.random.randint(0, size)
