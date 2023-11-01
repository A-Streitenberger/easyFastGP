from Base.individual import Individual

"""
Defines rules for inheriting selection classes
"""


class SelectionInterface:

    def execute(self, population: list[Individual]) -> list[Individual]:
        raise NotImplementedError
