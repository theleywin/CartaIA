from utils.kg.build_graph import build_directed_graph
from metaheuristic.learning_path import LearningPathProblem
from metaheuristic.ant_colony.aco import AntColony


def run_metaheuristic_test():    
    nodes_file = "nodes.csv"
    strong_edges_file = "strong_edges.csv"
    weak_edges_file = "weak_edges.csv"

    graph = build_directed_graph(nodes_file, strong_edges_file, weak_edges_file)
    problem = LearningPathProblem(
        graph=graph,
        known_topics={ "Breadth-first search"},
        weak_topics={},
        target_topic="Range Minimum Query"
    )

    ant_colony = AntColony(
        problem=problem,
    )
    solution, cost = ant_colony.run()
    print("Best solution:", solution)
    print("Cost:", cost)

