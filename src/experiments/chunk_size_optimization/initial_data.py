import random

def get_testing_chunk_sizes(amount: int) -> list[int]:
    return sorted(random.sample(range(128, 1025), amount))

db_topics = [
    "Maximum average subarray",
    "Finding polynomial roots modulo a prime",
    "Detecting negative cycles in graphs",
    "Solving the 15-puzzle with minimal moves",
    "Dynamic bridge detection in graphs",
    "Checking strong connectivity in a graph",
    "Randomized heap implementation",
    "Heavy-Light Decomposition (HLD)",
    "Determinant calculation using Gaussian elimination",
    "Efficient computation of all divisors",
    "Convex optimization in dynamic programming",
    "Segment intersection in 2D geometry",
    "Modular square root computation",
    "Efficient search in suffix trees",
    "Range Minimum Query (RMQ) structure and usage",
    "Minimum cost flow optimization",
    "Ternary search and its applications",
    "Cycle detection using Floyd's algorithm",
    "Lyndon factorization and its applications",
    "Manhattan distance calculation",
    "Comparison of Bellman-Ford and Dijkstra algorithms",
    "Rank of a matrix and how to compute it",
    "Tree painting algorithms",
    "Bipartite graph checking",
    "Montgomery modular arithmetic",
    "Fast Fourier Transform and polynomial multiplication",
    "Circle-line intersection algorithm",
    "Kirchhoff's theorem in graph theory",
    "Counting labeled trees",
    "String hashing techniques for long strings"
]
