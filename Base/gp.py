from Base.config.parameters import Parameters
from Base.individual import Individual
from Base.initializer import InitializerInterface
from Base.initializer_impl import TreeInitializerPostOrder
from Base.program import start_index_terminals, terminals, ephemeral_constant, \
    index_eph_constant, evaluate_postorder
from Evolution.evolution import Evolution
from Evolution.implementations.subtree_crossover import SubtreeCrossover
from Evolution.implementations.subtree_mutation import SubtreeMutation
from Evolution.implementations.symbol_regression_fitness import MSEFitness
from Evolution.implementations.tournament_selection import TournamentSelection
from Evolution.interfaces.crossover import CrossoverInterface
from Evolution.interfaces.fitness import FitnessInterface
from Evolution.interfaces.mutation import MutationInterface
from Evolution.interfaces.selection import SelectionInterface

"""
Inspired scikit-learn class for defining the components of a genetic program.
It contains the explicit implementation of initialisation, selection, crossover and so on.
"""


class GP():
    __slots__ = ["config", "initializer", "selection", "crossover", "mutation", "fitness_function", "best_last_gen"]

    def __init__(self):
        self.config = Parameters()
        self.initializer: InitializerInterface = TreeInitializerPostOrder(self.config)
        self.selection: SelectionInterface = TournamentSelection(self.config)
        self.crossover: CrossoverInterface = SubtreeCrossover(self.config)
        self.mutation: MutationInterface = SubtreeMutation(self.config)
        self.fitness_function: FitnessInterface = MSEFitness(self.config)
        self.best_last_gen: Individual = None

    def fit(self, X, y):
        self.__prepare_calculation(X, y)

        initial_population = self.__create_initial_population()

        # start evolution
        evolution = Evolution(self.config, self.crossover, self.mutation, self.selection, self.fitness_function)
        self.best_last_gen = evolution.execute(initial_population)

        return self

    def predict(self, X):
        if self.best_last_gen is None:
            raise Exception("The model must be trained before the prediction!")

        return evaluate_postorder(self.best_last_gen.program.nodes, self.X)

    def __prepare_calculation(self, X, y):
        self.__init_terminals_set(X)
        self.fitness_function.setup(X, y)

    def __init_terminals_set(self, X):
        if not terminals:
            for i in range(X.shape[1]):
                terminals.append(start_index_terminals + i)

            if ephemeral_constant:
                terminals.append(index_eph_constant)

    def __create_initial_population(self):
        initial_population = self.initializer.execute()
        fitness_values = self.fitness_function.evaluate_population(initial_population)

        for individual, fitness_value in zip(initial_population, fitness_values):
            individual.fitness = fitness_value

        return initial_population
