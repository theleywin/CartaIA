import re
import spacy
import csv
from collections import defaultdict
from itertools import combinations
from difflib import SequenceMatcher

nlp = spacy.load("en_core_sci_md")
INDEX_FILE = "src/utils/kg/index.txt"

def extract_entities(text):
    doc = nlp(text)
    return list({ent.text.strip() for ent in doc.ents if ent.text.strip()})

def is_chapter_line(line):
    return re.match(r"^\d+\s", line.strip()) is not None

def is_subsection_line(line):
    return re.match(r"^\d+(\.\d+)+\s", line.strip()) is not None

def parse_index(file_path):
    chapter = None
    chapter_entities = defaultdict(list)  # usamos lista para preservar orden
    line_entities = []

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if is_chapter_line(line):
                chapter = line
                ents = extract_entities(line)
                chapter_entities[chapter].extend(ents)
                line_entities.append((line, ents))

            elif is_subsection_line(line):
                ents = extract_entities(line)
                if chapter:
                    chapter_entities[chapter].extend(ents)
                line_entities.append((line, ents))

    return chapter_entities, line_entities

def string_similarity(a, b):
    """Permite detectar similitud parcial entre cadenas"""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()

def build_graph(chapter_entities, line_entities):
    nodes = set()
    strong_edges = set()
    weak_edges = set()

    for ents in chapter_entities.values():
        nodes.update(ents)
    for _, ents in line_entities:
        nodes.update(ents)

    # Relaciones fuertes dirigidas (por orden de aparición en el capítulo)
    for ents in chapter_entities.values():
        for i in range(len(ents) - 1):
            src = ents[i]
            dst = ents[i + 1]
            if src != dst:
                strong_edges.add((src, dst))  # dirigido: src → dst

    # Relaciones débiles más permisivas entre capítulos
    chapters = list(chapter_entities.items())
    for (ch1, ents1), (ch2, ents2) in combinations(chapters, 2):
        for e1 in ents1:
            for e2 in ents2:
                if string_similarity(e1, e2) > 0.6:
                    edge = (e1, e2)
                    reverse_edge = (e2, e1)
                    if edge not in strong_edges and reverse_edge not in strong_edges:
                        weak_edges.add(edge)

    return nodes, strong_edges, weak_edges

def classify_node_type(name):
    name = name.lower()
    if any(keyword in name for keyword in ["sort", "algorithm", "search", "matching"]):
        return "algorithm"
    elif any(keyword in name for keyword in ["tree", "heap", "queue", "stack", "hash", "graph"]):
        return "structure"
    else:
        return "other"

def save_nodes_with_attributes(nodes):
    with open("nodes.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "type", "difficulty", "estimated_time"])
        writer.writeheader()
        for node in sorted(nodes):
            node_type = classify_node_type(node)
            difficulty = min(10, max(1, len(node.split())))
            estimated_time = round(0.5 + difficulty * 0.5, 1)
            writer.writerow({
                "name": node,
                "type": node_type,
                "difficulty": difficulty,
                "estimated_time": estimated_time
            })

def save_edges(edges, filename, relation_type):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "target", "type"])
        for s, t in sorted(edges):
            writer.writerow([s, t, relation_type])

def main():
    chapter_entities, line_entities = parse_index(INDEX_FILE)
    nodes, strong_edges, weak_edges = build_graph(chapter_entities, line_entities)

    save_nodes_with_attributes(nodes)
    save_edges(strong_edges, "strong_edges.csv", "STRONG")
    save_edges(weak_edges, "weak_edges.csv", "WEAK")

    print(f"✅ Nodos: {len(nodes)}")
    print(f"✅ Relaciones fuertes: {len(strong_edges)}")
    print(f"✅ Relaciones débiles: {len(weak_edges)}")
    print("📦 Archivos generados: nodes.csv, strong_edges.csv, weak_edges.csv")

if __name__ == "__main__":
    main()