import numpy as np
from joblib import Parallel, delayed
from joblib.externals.loky import set_loky_pickler

from Base.config.parameters import Parameters
from Base.individual import Individual
from Base.program import evaluate_postorder
from Evolution.interfaces.fitness import FitnessInterface


class MSEFitness(FitnessInterface):
    __slots__ = ["config", "X", "y"]

    def __init__(self, config: Parameters):
        self.config = config
        self.X = None
        self.y = None
        self.evaluations = 0

    def setup(self, X, y):
        self.X = X
        self.y = y
        self.evaluations = 0

    def evaluate_population(self, population: list[Individual]):
        self.evaluations += self.config.pop_size

        set_loky_pickler('pickle')
        return Parallel(n_jobs=self.config.n_jobs)(
            delayed(self.evaluate)(individual)
            for individual in population)

    def evaluate(self, individual: Individual) -> float:
        if individual.fitness is not None:
            return individual.fitness

        output_root = evaluate_postorder(individual.program.nodes, self.X)

        fit_error = fitness_function(output_root, self.y)

        if np.isnan(fit_error):
            fit_error = np.inf

        return fit_error


def fitness_function(output_root, y_train):
    sqerrors = np.square(output_root - y_train)
    return np.sum(sqerrors) / len(y_train)
