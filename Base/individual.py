from Base.program import Program


class Individual:
    __slots__ = ["program", "fitness", "height"]

    def __init__(self, program: Program, fitness: float = None):
        self.program = program
        self.fitness = fitness
        self.height = None
