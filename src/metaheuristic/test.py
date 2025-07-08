from utils.kg.build_graph import build_directed_graph
from metaheuristic.learning_path import LearningPathProblem
from metaheuristic.ant_colony.aco import AntColony
from networkx import has_path

def run_metaheuristic_test():    
    nodes_file = "nodes.csv"
    strong_edges_file = "strong_edges.csv"
    weak_edges_file = "weak_edges.csv"

    graph = build_directed_graph(nodes_file, strong_edges_file, weak_edges_file)
    problem = LearningPathProblem(
        graph=graph,
        known_topics={"rooted-trees"},
        weak_topics={},
        target_topic="Red-Black-Trees"
    )

    ant_colony = AntColony(
        problem=problem,
        num_ants=80,
        iterations=100
    )
    print(has_path(graph, "Stacks-and-queues", "Maximum-Flow"))
    solution, cost = ant_colony.run()
    print("Best solution:", solution)
    print("Cost:", cost)

