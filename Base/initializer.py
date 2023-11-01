import enum

from Base.individual import Individual

"""
Defines rules for inheriting initializer
"""


class GrowType(enum.Enum):
    GROW = 'GROW'
    FULL = 'FULL'
    HALF_HALF = 'HALF_HALF'


class InitializerInterface:

    def execute(self) -> list[Individual]:
        raise NotImplementedError
