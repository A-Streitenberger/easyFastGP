import numpy as np

from Base.config.parameters import Parameters
from Base.individual import Individual
from Base.program import Program, get_rand_subtree, function_set
from Evolution.interfaces.crossover import CrossoverInterface


class SubtreeCrossover(CrossoverInterface):
    __slots__ = ["config"]

    def __init__(self, config: Parameters):
        self.config = config

    def execute(self, individual: Individual, donor: Program) -> Individual:
        """replaces random subtree of individual with donor.

        :param individual: The individual, which contains the program as an attribute
        :param donor: Corresponds to father individual
        :return: new Individual object with new program
        """
        start_m, end_m = get_rand_subtree(individual.program.nodes, function_set)
        start_f, end_f = get_rand_subtree(donor.nodes, function_set)

        new_pr = np.concatenate(
            (individual.program.nodes[:start_m], donor.nodes[start_f:end_f + 1],
             individual.program.nodes[end_m + 1:]))

        return Individual(Program(new_pr))
