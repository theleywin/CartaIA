import networkx as nx
import matplotlib.pyplot as plt

def build_knowledge_graph(nodes, strong_edges, weak_edges):
    G = nx.Graph()

    # Agregar nodos
    for node in nodes:
        G.add_node(node, type="Entity")

    # Agregar relaciones fuertes con atributo 'weight' o 'type'
    for source, target in strong_edges:
        G.add_edge(source, target, relation="STRONG", weight=2)

    # Agregar relaciones débiles
    for source, target in weak_edges:
        # Si ya existe una arista fuerte, no sobreescribimos
        if G.has_edge(source, target):
            continue
        G.add_edge(source, target, relation="WEAK", weight=1)

    return G

def visualize_graph(G):
    # Layout para posicionar nodos
    pos = nx.spring_layout(G, seed=42)

    # Extraemos aristas fuertes y débiles para pintarlas diferente
    strong_edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'STRONG']
    weak_edges = [(u, v) for u, v, d in G.edges(data=True) if d['relation'] == 'WEAK']

    plt.figure(figsize=(12, 10))
    # Dibujar nodos
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue')

    # Dibujar aristas fuertes en color rojo y más gruesas
    nx.draw_networkx_edges(G, pos, edgelist=strong_edges, width=2, edge_color='red')

    # Dibujar aristas débiles en color gris y más finas
    nx.draw_networkx_edges(G, pos, edgelist=weak_edges, width=1, edge_color='gray', style='dashed')

    # Etiquetas de nodos
    nx.draw_networkx_labels(G, pos, font_size=10, font_family="sans-serif")

    plt.title("Grafo de Conocimiento - NetworkX")
    plt.axis('off')
    plt.show()

# Ejemplo de uso con datos simulados
if __name__ == "__main__":
    # Ejemplo simple, reemplaza con tus datos reales
    nodes = {"Stacks", "Queues", "Binary Search Tree", "Insertion", "Strassen's algorithm"}
    strong_edges = {("Stacks", "Queues"), ("Stacks", "Binary Search Tree")}
    weak_edges = {("Insertion", "Binary Search Tree")}

    graph = build_knowledge_graph(nodes, strong_edges, weak_edges)
    visualize_graph(graph)