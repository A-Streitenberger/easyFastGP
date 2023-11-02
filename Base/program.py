import numpy as np
from numba import njit, types
from numba.typed.typeddict import Dict

"""
Definition of function and terminal set.
The numbers are representative for operations during evaluation.
"""

# definition of function / terminal ranges
start_index_functions = 90
start_index_terminals = 10
index_eph_constant = 8

# leaf nodes are initialized when training data is known in gp.py
terminals = []
ephemeral_constant = True
lower_limit_constants = -0.995
upper_limit_constants = 0.997

# function_set as numba dictionary
function_set = Dict.empty(
    key_type=types.float64,
    value_type=types.int64,
)

# add
function_set[90] = 2
# sub
function_set[91] = 2
# mult
function_set[92] = 2
# div
function_set[93] = 2


class Program:
    __slots__ = ["nodes"]

    def __init__(self, nodes: []):
        self.nodes = np.array(nodes)


@njit(cache=True)
def get_height(nodes, func_dict):
    stack = [0]
    max_depth = 0
    for i in range(len(nodes) - 1, -1, -1):
        elem = nodes[i]
        depth = stack.pop()
        max_depth = max(max_depth, depth)

        if elem >= start_index_functions:
            arity = func_dict[elem]
            for j in range(arity):
                stack.append(depth + 1)

    return max_depth


@njit(cache=True)
def get_custom_height(nodes, index, func_dict):
    stack = [0]
    max_depth = 0
    for i in range(len(nodes) - 1, -1, -1):
        if i == index:
            return stack.pop()

        elem = nodes[i]
        depth = stack.pop()
        max_depth = max(max_depth, depth)

        if elem >= start_index_functions:
            arity = func_dict[elem]
            for j in range(arity):
                stack.append(depth + 1)

    return max_depth


@njit(cache=True)
def evaluate_postorder(nodes, X):
    params = []

    for i in range(nodes.shape[0]):
        node = nodes[i]

        if node == 90:
            n1 = params.pop()
            n2 = params.pop()
            params.append(np.add(n2, n1))
        elif node == 91:
            n1 = params.pop()
            n2 = params.pop()
            params.append(np.subtract(n2, n1))
        elif node == 92:
            n1 = params.pop()
            n2 = params.pop()
            params.append(np.multiply(n2, n1))
        elif node == 93:
            n1 = params.pop()
            n2 = params.pop()
            sign_X1 = np.sign(n1)
            sign_X1[sign_X1 == 0] = 1
            params.append(np.multiply(sign_X1, n2) / (1e-6 + np.abs(n1)))
        elif start_index_terminals <= node < start_index_functions:
            params.append(np.ascontiguousarray(X[:, int(node % start_index_terminals)]))
        else:
            params.append(np.ascontiguousarray(np.full(X.shape[0], node)))

    return params.pop()


# function_set 90%, terminals 10%
@njit(cache=True)
def get_rand_subtree(nodes, func_dict) -> [int, int]:
    prob = np.where(nodes >= start_index_functions, 0.9, 0.1)
    prob = np.cumsum(prob / np.sum(prob))
    end = np.searchsorted(prob, np.random.uniform(0.0, 1.0))

    stack = 1
    start = end

    while stack > end - start:
        node = nodes[start]
        if node >= start_index_functions:
            stack += func_dict[node]
        start -= 1

    return start + 1, end


# function_set 90%, terminals 10%
@njit(cache=True)
def get_rand_node(nodes) -> int:
    probs = np.where(nodes >= start_index_functions, 0.9, 0.1)
    probs = np.cumsum(probs / np.sum(probs))
    return np.searchsorted(probs, np.random.uniform(0.0, 1.0))
