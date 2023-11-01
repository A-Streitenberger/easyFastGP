from operator import attrgetter

import numpy as np

from Base.config.parameters import Parameters
from Base.individual import Individual
from Evolution.interfaces.selection import SelectionInterface


class TournamentSelection(SelectionInterface):
    __slots__ = ["config"]

    def __init__(self, config: Parameters):
        self.config = config

    def execute(self, population: list[Individual]) -> list[Individual]:
        """
        Performs tournament selection on the given population to create a new population of individuals.

        :param population: input population
        :return: new population of selected individuals
        """

        selection: list[Individual] = []

        while len(selection) < self.config.pop_size:
            choices = []
            for _ in range(self.config.tournament_size):
                choices.append(population[np.random.randint(0, self.config.pop_size)])
            selection.append(min(choices, key=attrgetter("fitness")))

        return selection
