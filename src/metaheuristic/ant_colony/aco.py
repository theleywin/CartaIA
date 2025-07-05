from typing import Generic, List, Tuple, TypeVar
from metaheuristic.ant_colony.problem import Problem
from metaheuristic.ant_colony.ant import Ant
from metaheuristic.ant_colony.pheromones import PheromoneTable

State = TypeVar('State')

class AntColony(Generic[State]):
    def __init__(
        self,
        problem: Problem[State],
        num_ants: int = 10,
        alpha: float = 1.0,
        beta: float = 2.0,
        rho: float = 0.1,
        iterations: int = 100
    ):
        self.problem = problem
        self.pheromones = PheromoneTable[State]()
        self.num_ants = num_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.iterations = iterations
        self.best_solution: List[State] = []
        self.best_cost: float = float("inf")

    def run(self) -> Tuple[List[State], float]:
        all_edges = [(u, v) for u in self.problem.start_nodes() for v in self.problem.valid_moves(u, set())]
        self.pheromones.initialize(all_edges)

        for _ in range(self.iterations):
            ants = [Ant(self.problem, self.pheromones, self.alpha, self.beta) for _ in range(self.num_ants)]
            solutions = [ant.run() for ant in ants]
            costs = [self.problem.evaluate(s) for s in solutions]

            for sol, cost in zip(solutions, costs):
                if cost < self.best_cost:
                    self.best_cost = cost
                    self.best_solution = sol

            self.pheromones.evaporate(self.rho)
            for sol, cost in zip(solutions, costs):
                if cost > 0:
                    self.pheromones.reinforce(sol, 1.0 / cost)