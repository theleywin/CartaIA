import networkx as nx
from typing import List, Set
from metaheuristic.ant_colony.problem import Problem

class LearningPathProblem(Problem[str]):
    def __init__(
        self,
        graph: nx.DiGraph,
        known_topics: Set[str],
        weak_topics: Set[str],
        target_topic: str
    ):
        self.graph = graph
        self.known_topics = known_topics
        self.weak_topics = weak_topics
        self.target_topic = target_topic

    def start_nodes(self) -> List[str]:
        return list(self.known_topics)

    def is_terminal(self, node: str) -> bool:
        return node == self.target_topic

    def valid_moves(self, current: str, visited: Set[str]) -> List[str]:
        return [n for n in self.graph.successors(current) if n not in visited]

    def heuristic(self, i: str, j: str) -> float:
        edge_type = self.graph.edges[i, j].get("type", "STRONG")
        weight = 1.0 if edge_type == "STRONG" else 0.5

        difficulty = self.graph.nodes[j].get("difficulty", 1.0)
        time = self.graph.nodes[j].get("estimated_time", 1.0)

        reinforcement = 1.5 if j in self.weak_topics else 1.0

        return weight * reinforcement / (difficulty * time)

    def evaluate(self, solution: List[str]) -> float:
        total_time = 0.0
        total_difficulty = 0.0
        for node in solution:
            attrs = self.graph.nodes[node]
            total_time += attrs.get("estimated_time", 1.0)
            total_difficulty += attrs.get("difficulty", 1.0)
        return total_time + total_difficulty
