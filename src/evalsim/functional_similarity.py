"""NLI-based entailment scoring for functional similarity."""

import numpy as np

from askme.rtp.nli import NLIWithChunkingAndPooling


def all_entailment_scores(
    hypotheses: list[str],
    premises: list[str],
    model: NLIWithChunkingAndPooling,
) -> np.ndarray:
    """Run NLI for every (hypothesis, premise) pair.

    Returns an (n_hypotheses, n_premises) matrix of entailment scores.
    """
    scores = []
    for h in hypotheses:
        results = model(premises, h)
        scores.append([r.entailment_score for r in results])
    return np.array(scores)
