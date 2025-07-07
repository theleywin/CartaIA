import random
from typing import Generic, List, Set, TypeVar
from metaheuristic.ant_colony.problem import Problem
from metaheuristic.ant_colony.pheromones import PheromoneTable

State = TypeVar('State')

class Ant(Generic[State]):
    def __init__(self, problem: Problem[State], pheromones: PheromoneTable[State], alpha: float = 1.0, beta: float = 2.0):
        self.problem = problem
        self.pheromones = pheromones
        self.alpha = alpha
        self.beta = beta

    def run(self) -> List[State]:
        path: List[State] = []
        visited: Set[State] = set()
        current = random.choice(self.problem.start_nodes())
        path.append(current)
        visited.add(current)

        while not self.problem.is_terminal(current):
            candidates = self.problem.valid_moves(current, visited)
            if not candidates:
                break
            probabilities = self._compute_probabilities(current, candidates)
            next_node = random.choices(candidates, weights=probabilities, k=1)[0]
            path.append(next_node)
            visited.add(next_node)
            current = next_node
        return path

    def _compute_probabilities(self, current: State, candidates: List[State]) -> List[float]:
        total = 0.0
        values: List[float] = []
        for j in candidates:
            tau = self.pheromones[current, j] ** self.alpha
            eta = self.problem.heuristic(current, j) ** self.beta
            val = (tau + 1e-10) * (eta + 1e-10)
            values.append(val)
            total += val

        return [v / total for v in values]
