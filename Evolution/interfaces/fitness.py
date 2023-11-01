from Base.individual import Individual

"""
Defines rules for inheriting fitness classes
"""


class FitnessInterface:

    def setup(self, X, y, X_train, X_test, y_train, y_test):
        raise NotImplementedError

    def evaluate_population(self, population: list[Individual]) -> [float]:
        raise NotImplementedError

    def evaluate(self, individual: Individual) -> float:
        raise NotImplementedError
