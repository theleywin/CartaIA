def is_relevant_cosine(score: float) -> bool:
    threshold = 0.3
    return score > threshold

def is_relevant_l2(score: float) -> bool:
    threshold = 1.2
    return score < threshold