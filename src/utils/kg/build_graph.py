import networkx as nx
import csv
import matplotlib.pyplot as plt

def load_nodes(filepath: str) -> list:
    nodes = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            node_attrs = {
                "name": row["name"],
                "type": row.get("type", ""),
                "difficulty": float(row.get("difficulty", 1.0)),
                "estimated_time": float(row.get("estimated_time", 1.0))
            }
            nodes.append((row["name"], node_attrs))
    return nodes

def load_edges(filepath: str) -> list:
    edges = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row["source"]
            target = row["target"]
            edge_type = row.get("type", "")
            edges.append((source, target, {"type": edge_type}))
    return edges

def build_directed_graph(nodes_file: str, strong_edges_file: str, weak_edges_file: str) -> nx.DiGraph:
    graph = nx.DiGraph()

    # Agregar nodos con atributos
    nodes = load_nodes(nodes_file)
    for node_id, node_attrs in nodes:
        graph.add_node(node_id, **node_attrs)

    # Agregar relaciones fuertes y débiles
    strong_edges = load_edges(strong_edges_file)
    weak_edges = load_edges(weak_edges_file)

    graph.add_edges_from(strong_edges)
    graph.add_edges_from(weak_edges)

    return graph

def visualize_directed_graph(graph: nx.DiGraph):
    plt.figure(figsize=(14, 10))

    pos = nx.spring_layout(graph, seed=10, k=0.15, iterations=50)

    # Dibuja nodos
    nx.draw_networkx_nodes(graph, pos, node_color='lightblue', node_size=100)

   
    nx.draw_networkx_labels(graph, pos, font_size=6)

    # Aristas fuertes (rojo) y débiles (azul punteado)
    strong_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "STRONG"]
    weak_edges = [(u, v) for u, v, d in graph.edges(data=True) if d.get("type") == "WEAK"]

    nx.draw_networkx_edges(graph, pos, edgelist=strong_edges, edge_color="red", arrows=True, arrowsize=10, width=1)
    nx.draw_networkx_edges(graph, pos, edgelist=weak_edges, edge_color="blue", style="dashed", arrows=True, arrowsize=10, width=1)

    plt.title("📚 Grafo dirigido de conocimiento", fontsize=12)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    nodes_file = "nodes.csv"
    strong_edges_file = "strong_edges.csv"
    weak_edges_file = "weak_edges.csv"

    graph = build_directed_graph(nodes_file, strong_edges_file, weak_edges_file)
    visualize_directed_graph(graph)