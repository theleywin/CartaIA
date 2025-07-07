# base.py
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List, Set

State = TypeVar('State')  # Puede ser int, str, etc.

class Problem(ABC, Generic[State]):
    @abstractmethod
    def valid_moves(self, current: State, visited: Set[State]) -> List[State]: ...

    @abstractmethod
    def heuristic(self, i: State, j: State) -> float: ...

    @abstractmethod
    def evaluate(self, solution: List[State]) -> float: ...

    @abstractmethod
    def start_nodes(self) -> List[State]: ...

    @abstractmethod
    def is_terminal(self, node: State) -> bool: ...
