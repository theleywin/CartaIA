COSINE_THRESHOLD = 0.3
L2_THRESHOLD = 1.2

def is_relevant_cosine(score: float, threshold: float = COSINE_THRESHOLD) -> bool:
    return score > threshold

def is_relevant_l2(score: float, threshold: float = L2_THRESHOLD) -> bool:
    return score < threshold