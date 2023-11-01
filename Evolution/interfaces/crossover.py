from Base.individual import Individual
from Base.program import Program

"""
Defines rules for inheriting crossover classes
"""


class CrossoverInterface:

    def execute(self, individual: Individual, donor: Program) -> Individual:
        raise NotImplementedError
