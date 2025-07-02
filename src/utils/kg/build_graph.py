import networkx as nx
import csv
import matplotlib.pyplot as plt

def load_nodes(filepath):
    nodes = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            nodes.append(row["name"])
    return nodes

def load_edges(filepath):
    edges = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            source = row["source"]
            target = row["target"]
            edge_type = row.get("type", "")
            edges.append((source, target, {"type": edge_type}))
    return edges

def build_directed_graph(nodes_file, strong_edges_file, weak_edges_file):
    G = nx.DiGraph()

    # Agregar nodos
    nodes = load_nodes(nodes_file)
    G.add_nodes_from(nodes)

    # Agregar relaciones fuertes y débiles (aristas dirigidas)
    strong_edges = load_edges(strong_edges_file)
    weak_edges = load_edges(weak_edges_file)

    G.add_edges_from(strong_edges)
    G.add_edges_from(weak_edges)

    return G
def visualize_directed_graph(G):
    plt.figure(figsize=(12, 8))

    pos = nx.spring_layout(G, seed=10)  # Posiciones de los nodos (puedes usar otros layouts)

    # Dibuja nodos
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=200)

    # Etiquetas de nodos
    nx.draw_networkx_labels(G, pos, font_size=6, font_weight='bold')

    # Separa aristas por tipo para colorear diferente
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'STRONG']
    weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == 'WEAK']

    # Dibuja aristas fuertes en rojo, con flechas sólidas
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, edge_color='red', arrows=True, arrowsize=10, width=2)

    # Dibuja aristas débiles en azul, con flechas punteadas
    nx.draw_networkx_edges(G, pos, edgelist=weak_edges, edge_color='blue', style='dashed', arrows=True, arrowsize=20, width=1)

    plt.title("Grafo dirigido de entidades y relaciones")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    nodes_file = "nodes.csv"
    strong_edges_file = "strong_edges.csv"
    weak_edges_file = "weak_edges.csv"

    graph = build_directed_graph(nodes_file, strong_edges_file, weak_edges_file)
    visualize_directed_graph(graph)