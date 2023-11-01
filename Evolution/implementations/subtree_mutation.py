import numpy as np

from Base.individual import Individual
from Base.initializer_impl import TreeInitializerPostOrder
from Base.program import get_rand_subtree, Program, function_set
from Evolution.interfaces.mutation import MutationInterface


class SubtreeMutation(MutationInterface):
    __slots__ = ["config"]

    def __init__(self, config):
        self.config = config
        self.treeInitializer = TreeInitializerPostOrder(config)

    def execute(self, individual: Individual) -> Individual:
        """Mutates an entire subtree in the program. This is first created with the initializer and corresponds to
        the properties of the trees from the original generation (with minimum and maximum size). The new
        subtree is then indirectly replaced by a random subtree in the program.

        :param individual: The individual, which contains the program as an attribute
        :return: new Individual object with mutated program
        """

        new_sub = self.treeInitializer.build_one_program()
        start, end = get_rand_subtree(individual.program.nodes, function_set)

        new_pr = np.concatenate(
            (individual.program.nodes[:start], new_sub.nodes,
             individual.program.nodes[end + 1:]))

        return Individual(Program(new_pr))
