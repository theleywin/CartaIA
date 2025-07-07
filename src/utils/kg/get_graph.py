import csv
import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from matplotlib import pyplot as plt
from utils.embedding_loader import llm_loader
from utils.kg.schemas import TopicMetadata
import networkx as nx
from typing import Dict, Any

MD_FOLDER = "data/algoritmos/md"

def extract_title_and_content(filepath: str):
    title = None
    content_lines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            content_lines.append(line)
            if title is None:
                match = re.match(r"#\s+(.+)", line)
                if match:
                    title = match.group(1).strip()

    return title, "".join(content_lines)

def load_all_md_documents(md_folder: str = MD_FOLDER):
    title_by_file = {}
    content_by_file = {}

    for filename in os.listdir(md_folder):
        if filename.endswith(".md"):
            path = os.path.join(md_folder, filename)
            title, content = extract_title_and_content(path)

            if not title:
                print(f"⚠️  No se encontró título en: {filename}")
                title = filename.replace(".md", "")

            title_by_file[filename] = title
            content_by_file[filename] = content

    return title_by_file, content_by_file

def infer_metadata_for_topic(llm: ChatGoogleGenerativeAI, content: str) -> TopicMetadata:
    chain = llm.with_structured_output(TopicMetadata)

    prompt = f"""
    Eres un experto en estructuras de datos y algoritmos. Lee el siguiente artículo técnico y determina:

    - El tipo de contenido: uno de ALGORITHM, STRUCT, THEORY, PROBLEM, OTHER.
    - La dificultad del contenido del 1.0 (muy fácil) al 5.0 (muy difícil).
    - El tiempo estimado en horas para que un estudiante de nivel intermedio entienda este contenido.

    Artículo:
    \"\"\"
    {content}
    \"\"\"
    """

    try:
        result = chain.invoke(prompt)
        result = result.model_dump(mode="json", exclude_none=True)
        return result
    except Exception as e:
        print(f"❌ Error en inferencia del LLM: {e}")
        return None

def build_knowledge_graph(
    title_by_file: Dict[str, str],
    content_by_file: Dict[str, str],
    attrs_by_file: Dict[str, Dict[str, Any]]
) -> nx.DiGraph:
    graph = nx.DiGraph()

    for filename, title in title_by_file.items():
        attrs = attrs_by_file.get(filename, {})
        graph.add_node(
            title,
            type=attrs.get("type", "OTHER"),
            difficulty=attrs.get("difficulty", 1.0),
            estimated_time=attrs.get("estimated_time", 1.0)
        )

    for source_file, content in content_by_file.items():
        source_title = title_by_file[source_file]

        linked_files = re.findall(r"\]\((.+?\.md)\)", content)

        for raw_link in linked_files:
            linked_filename = os.path.basename(raw_link)

            if linked_filename in title_by_file:
                target_title = title_by_file[linked_filename]
                graph.add_edge(target_title, source_title, type="STRONG")
    return graph

def save_graph_to_csv(graph: nx.DiGraph, nodes_filepath: str, edges_filepath: str):
    # Guardar nodos
    with open(nodes_filepath, "w", newline="", encoding="utf-8") as f_nodes:
        fieldnames = ["name", "type", "difficulty", "estimated_time"]
        writer = csv.DictWriter(f_nodes, fieldnames=fieldnames)
        writer.writeheader()
        for node, data in graph.nodes(data=True):
            writer.writerow({
                "name": node,
                "type": data.get("type", "OTHER"),
                "difficulty": data.get("difficulty", 1.0),
                "estimated_time": data.get("estimated_time", 1.0)
            })

    # Guardar aristas fuertes
    with open(edges_filepath, "w", newline="", encoding="utf-8") as f_edges:
        fieldnames = ["source", "target", "type"]
        writer = csv.DictWriter(f_edges, fieldnames=fieldnames)
        writer.writeheader()
        for u, v, data in graph.edges(data=True):
            if data.get("type") == "STRONG":
                writer.writerow({
                    "source": u,
                    "target": v,
                    "type": "STRONG"
                })

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
    plt.savefig("knowledge_graph.png", dpi=300)

def create_knowledge_graph():
    MD_FOLDER = "data/algoritmos/md"
    NODES_CSV = "nodes.csv"
    EDGES_CSV = "strong_edges.csv"

    print("Cargando archivos markdown...")
    title_by_file, content_by_file = load_all_md_documents(MD_FOLDER)

    print("Cargando LLM...")
    llm = llm_loader()
    if llm is None:
        print("No se pudo cargar LLM, saliendo.")
        return

    print("Inferiendo metadatos para cada documento...")
    attrs_by_file = {}
    for filename, content in content_by_file.items():
        print(f"Procesando {filename}...")
        attrs_by_file[filename] = infer_metadata_for_topic(llm, content)
        

    print("Construyendo grafo de conocimiento...")
    graph = build_knowledge_graph(title_by_file, content_by_file, attrs_by_file)

    print("Guardando grafo en CSV...")
    save_graph_to_csv(graph, NODES_CSV, EDGES_CSV)

    print("¡Proceso completado!")