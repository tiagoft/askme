import torch.nn as nn
import torch
from askme.rtp.nli import NLIWithChunkingAndPooling
import numpy as np



def logical_similarity(
    a : str,
    b : str,
    model : NLIWithChunkingAndPooling,
) -> float:
    """Calculate the logical similarity between two texts using a pre-trained model."""
    ent_ab = model([a], b)
    ent_ba = model([b], a)
    return np.sqrt(ent_ab[0].entailment_score * ent_ba[0].entailment_score)


def pairwise_logical_similarity(
    texts: list[str],
    model: NLIWithChunkingAndPooling,
) -> np.ndarray:
    """Calculate the pairwise logical similarity between two texts using a pre-trained model."""
    similarities = np.zeros((len(texts), len(texts)))
    for i in range(len(texts)):
        for j in range(i + 1, len(texts)):
            similarities[i, j] = logical_similarity(texts[i], texts[j], model)
            similarities[j, i] = similarities[i, j]  # Symmetric matrix
    return similarities