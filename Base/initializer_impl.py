import numpy as np
from numba import njit
from numpy.random import randint

from Base.config.parameters import Parameters
from Base.individual import Individual
from Base.initializer import InitializerInterface, GrowType
from Base.program import Program, start_index_functions, terminals, function_set, \
    index_eph_constant, lower_limit_constants, upper_limit_constants


class TreeInitializerPostOrder(InitializerInterface):
    __slots__ = ["config"]

    def __init__(self, config: Parameters):
        self.config = config

    def execute(self) -> list[Individual]:
        pop = []

        for i in range(self.config.pop_size):
            program = self.build_one_program()
            pop.append(Individual(program))

        return pop

    def build_one_program(self) -> Program:
        height = randint(self.config.min_height, self.config.initialization_max_tree_height)

        grow_method = self.config.initialization_method
        if grow_method == GrowType.HALF_HALF:
            grow_method = GrowType.GROW if np.random.random() < 0.5 else GrowType.FULL

        tree = self.__generate_tree(list(function_set.keys()), terminals, height, self.config.min_height,
                                    method=grow_method)

        return Program(self.flatten_array(tree))

    def __generate_tree(self, function_arr, terminal_arr, target_height, min_height, curr_height=0,
                        method=GrowType.GROW) -> Program:

        if curr_height >= target_height:
            node = terminal_arr[randint(0, len(terminal_arr))]
        elif method == GrowType.FULL or (method == GrowType.GROW and curr_height < min_height):
            node = function_arr[randint(0, len(function_arr))]
        else:
            t_f = terminal_arr + function_arr
            node = t_f[randint(0, len(t_f))]

        arity = 0
        if node >= start_index_functions:
            arity = function_set[node]
        elif node == index_eph_constant:
            node = np.random.uniform(lower_limit_constants, upper_limit_constants)

        children = []
        for i in range(arity):
            child = self.__generate_tree(function_arr, terminal_arr, target_height, min_height,
                                         curr_height=curr_height + 1, method=method)
            children.append(child)

        children.append(node)
        return children

    # __generate_tree creates a tree with nested arrays, it must be flattened
    def flatten_array(self, arr):
        result = []
        for i in arr:
            if isinstance(i, list):
                result.extend(self.flatten_array(i))
            else:
                result.append(i)
        return result


@njit(cache=True)
def randint(n1, n2):
    return np.random.randint(n1, n2)
