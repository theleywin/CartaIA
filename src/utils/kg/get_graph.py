import re
import spacy
from collections import defaultdict
import csv

# Carga el modelo scispacy (asegúrate que esté instalado)
nlp = spacy.load("en_core_sci_md")

# Archivo de índice (puedes cambiarlo a tu archivo real)
INDEX_FILE = "src/utils/kg/index.txt"

def extract_entities(text):
    """Extrae entidades nombradas del texto usando scispacy"""
    doc = nlp(text)
    return list({ent.text.strip() for ent in doc.ents if ent.text.strip()})

def is_chapter_line(line):
    """Detecta si una línea es título de capítulo: empieza con número entero y espacio, ej: '4 Getting Started'"""
    return re.match(r"^\d+\s", line.strip()) is not None

def is_subsection_line(line):
    """Detecta si es línea de subsección (ej: '4.2 Strassens algorithm for matrix multiplication')"""
    return re.match(r"^\d+(\.\d+)+\s", line.strip()) is not None

def parse_index(file_path: str) -> tuple[defaultdict[set], list[tuple[str, list[str]]]]:
    """
    Procesa el índice completo y extrae:
    - capítulos con sus entidades
    - líneas subsección con sus entidades
    """
    chapter = None
    chapter_entities = defaultdict(set)  # {capítulo: set(entidades)}
    line_entities = []  # [(line_text, [entidades])]

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_chapter_line(line):
                chapter = line
                ents = extract_entities(line)
                chapter_entities[chapter].update(ents)
                # Guarda la línea como sección también
                line_entities.append((line, ents))

            elif is_subsection_line(line):
                ents = extract_entities(line)
                # Agrega estas entidades al capítulo actual
                if chapter:
                    chapter_entities[chapter].update(ents)
                line_entities.append((line, ents))
            else:
                # Líneas que no son ni capítulos ni subsecciones se ignoran
                continue

    return chapter_entities, line_entities

def build_graph(chapter_entities: defaultdict[set], line_entities: list[tuple[str, list[str]]]) -> tuple[set, set, set]:
    """
    Construye listas para nodos y relaciones para Neo4j.
    Relaciones fuertes: entre todas las entidades del mismo capítulo.
    Relaciones débiles: entre entidades de la misma línea.
    """
    nodes = set()
    strong_edges = set()
    weak_edges = set()

    # Añadir nodos
    for ents in chapter_entities.values():
        nodes.update(ents)
    for _, ents in line_entities:
        nodes.update(ents)

    # Relaciones fuertes (capítulo): Conectar todas las entidades del capítulo entre sí
    for ents in chapter_entities.values():
        ents_list = list(ents)
        for i in range(len(ents_list)):
            for j in range(i + 1, len(ents_list)):
                edge = tuple(sorted((ents_list[i], ents_list[j])))
                strong_edges.add(edge)

    # Relaciones débiles (línea): Conectar todas entidades en la misma línea
    for _, ents in line_entities:
        ents_list = list(ents)
        for i in range(len(ents_list)):
            for j in range(i + 1, len(ents_list)):
                edge = tuple(sorted((ents_list[i], ents_list[j])))
                # Evitar duplicados y no sobreescribir relaciones fuertes
                if edge not in strong_edges:
                    weak_edges.add(edge)

    return nodes, strong_edges, weak_edges

def save_to_csv(nodes: set, strong_edges: set, weak_edges: set):
    # Guardar nodos
    with open("nodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["name"])
        for n in sorted(nodes):
            writer.writerow([n])

    # Guardar relaciones fuertes
    with open("strong_edges.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "type"])
        for s, t in sorted(strong_edges):
            writer.writerow([s, t, "STRONG"])

    # Guardar relaciones débiles
    with open("weak_edges.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "type"])
        for s, t in sorted(weak_edges):
            writer.writerow([s, t, "WEAK"])

def main():
    chapter_entities, line_entities = parse_index(INDEX_FILE)
    nodes, strong_edges, weak_edges = build_graph(chapter_entities, line_entities)
    save_to_csv(nodes, strong_edges, weak_edges)
    print(f"Nodos: {len(nodes)}")
    print(f"Relaciones fuertes: {len(strong_edges)}")
    print(f"Relaciones débiles: {len(weak_edges)}")
    print("CSV generados: nodes.csv, strong_edges.csv, weak_edges.csv")

if __name__ == "__main__":
    main()