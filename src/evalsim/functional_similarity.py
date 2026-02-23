from askme.rtp.nli import NLIWithChunkingAndPooling
from .commons import average_binary_jensen_shannon_similarity
import numpy as np 


def all_entailment_scores(
    hypotheses : list[str],
    premises : list[str],
    model : NLIWithChunkingAndPooling,
) -> np.ndarray:
    """Calculate the functional similarity between two lists of hypotheses, given a dataset of premises.
    A high functional similarity means that the hypotheses evaluate the same to all premises."""

    scores = []
    for h in hypotheses:
        results = model(premises, h)
        these_scores = [r.entailment_score for r in results]
        scores.append(these_scores)
    scores = np.array(scores)
    
    return scores

def pairwise_functional_similarity(
    scores : np.ndarray,
) -> np.ndarray:
    """Calculate the pairwise functional similarity given the scores given to each of them.
    A high functional similarity means that the hypotheses evaluate the same to all premises."""

    n_hypotheses = scores.shape[0]
    
    similarities = np.zeros((n_hypotheses, n_hypotheses))
    for i in range(n_hypotheses):
        for j in range(i + 1, n_hypotheses):
            similarities[i, j] = average_binary_jensen_shannon_similarity(scores[i], scores[j])
            similarities[j, i] = similarities[i, j]  # Symmetric matrix
    return similarities