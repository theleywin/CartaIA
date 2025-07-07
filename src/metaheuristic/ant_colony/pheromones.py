from typing import Dict, Tuple, TypeVar, Generic, List

State = TypeVar('State')

class PheromoneTable(Generic[State]):
    def __init__(self, initial: float = 1.0):
        self.values: Dict[Tuple[State, State], float] = {}
        self.initial = initial

    def initialize(self, edges: List[Tuple[State, State]]) -> None:
        for edge in edges:
            self[edge] = self.initial

    def evaporate(self, rho: float) -> None:
        for key in self.values:
            self[key] *= (1.0 - rho)

    def reinforce(self, path: List[State], delta: float) -> None:
        for i in range(len(path) - 1):
            edge = (path[i], path[i + 1])
            self[edge] += delta

    def __getitem__(self, edge: Tuple[State, State]) -> float:
        return self.values.get(edge, self.initial)
    
    def __setitem__(self, edge: Tuple[State, State], value: float) -> None:
        self.values[edge] = value