from Base.individual import Individual

"""
Defines rules for inheriting mutation classes
"""


class MutationInterface:

    def execute(self, individual: Individual) -> Individual:
        raise NotImplementedError
