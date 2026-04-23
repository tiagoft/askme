from evalsim.lexical_similarity import pairwise_jaccard_ngram_similarity
from evalsim.semantic_similarity import pairwise_cosine_similarity
from evalsim.logical_similarity import pairwise_logical_similarity
from evalsim.functional_similarity import all_entailment_scores, pairwise_functional_similarity
from askme.rtp.nli import NLIWithChunkingAndPooling
from sentence_transformers import SentenceTransformer


def test_pairwise_jaccard_ngram_similarity():
    texts = ["the cat is on the roof", "the cat is on the roof", "the dog is in the yard"]
    similarities = pairwise_jaccard_ngram_similarity(texts, n=2)
    assert similarities[0, 1] == 1.0
    assert similarities[0, 2] < 1.0
    assert similarities[1, 2] < 1.0
    
def test_pairwise_cosine_similarity():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = ["the cat is on the roof", "the cat is on the roof", "the dog is in the yard"]
    similarities = pairwise_cosine_similarity(texts, model)
    assert similarities[0, 1] > 0.99
    assert similarities[0, 2] < 0.99
    assert similarities[1, 2] < 0.99
    
def test_pairwise_logical_similarity():
    model = NLIWithChunkingAndPooling()
    
    texts = ["the cat is on the roof", "the cat is on the roof", "the dog is in the yard"]
    similarities = pairwise_logical_similarity(texts, model)
    assert similarities[0, 1] > 0.6
    assert similarities[0, 2] < 0.6
    assert similarities[1, 2] < 0.6
    
def test_pairwise_functional_similarity():
    model = NLIWithChunkingAndPooling()

    # h0 is entailed by premise 0; h1 is entailed by premises 1 and 2.
    # h2 is not entailed by any premise (cat in the yard — wrong location).
    # Note: "a feline is on the roof" is omitted because the NLI model does
    # not reliably map feline→cat, so it cannot be used as a reliable premise.
    premises = [
        "the cat is on the roof",
        "the dog is in the yard",
        "a canine is in the yard",
    ]
    hypotheses = [
        "the cat is on the roof",   # h0
        "the dog is in the yard",   # h1
        "the cat is in the yard",   # h2 — constant (never entailed)
    ]

    scores = all_entailment_scores(hypotheses, premises, model)
    similarities = pairwise_functional_similarity(scores)

    # h0 and h1 produce perfectly anti-correlated binary patterns →
    # NMI = 1 (knowing one perfectly predicts the other).
    assert similarities[0, 1] > 0.5

    # h2 produces a constant (all-zero) pattern → NMI = 0 with any other.
    assert similarities[0, 2] < 0.5
    assert similarities[1, 2] < 0.5