from Base.initializer import GrowType

"""
Central configuration class contains used parameters
"""


class Parameters:

    def __init__(self):
        self.pop_size = 500

        # initialization
        self.initialization_method: GrowType = GrowType.HALF_HALF
        self.min_height = 2
        self.initialization_max_tree_height = 6

        # selection
        self.tournament_size = 7
        self.elitism = 0

        # probabilities
        self.crossover_rate = 0.90
        self.mutation_rate = 0.03

        # other evolution params
        # -1 when not needed
        self.max_size = -1
        self.max_height = 90

        # run configurations
        # -1 when not needed
        self.max_generations = 500
        self.max_time = -1

        # jobs for parallelization
        self.n_jobs = 4
