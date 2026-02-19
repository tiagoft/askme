def jaccard_similarity(set1, set2):
    """
    Calculate the Jaccard similarity between two sets.

    Args:
        set1 (set): The first set.
        set2 (set): The second set.

    Returns:
        float: The Jaccard similarity coefficient.
    """
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0  # Avoid division by zero
    
    return intersection / union

def cosine_similarity(vec1, vec2):
    """
    Calculate the Cosine similarity between two vectors.

    Args:
        vec1 (list): The first vector.
        vec2 (list): The second vector.

    Returns:
        float: The Cosine similarity coefficient.
    """
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude_vec1 = sum(a ** 2 for a in vec1) ** 0.5
    magnitude_vec2 = sum(b ** 2 for b in vec2) ** 0.5
    
    if magnitude_vec1 == 0 or magnitude_vec2 == 0:
        return 0.0  # Avoid division by zero
    
    return dot_product / (magnitude_vec1 * magnitude_vec2)