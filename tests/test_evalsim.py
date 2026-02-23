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
    
def test_parwise_functional_similarity():
    model = NLIWithChunkingAndPooling()
    
    premises = ["the cat is on the roof", "the dog is in the yard", "a canine is in the yard", "a feline is on the roof"]
    hypotheses = ["the cat is on the roof", "the dog is in the yard", "the cat is in the yard"]
    
    scores = all_entailment_scores(hypotheses, premises, model)
    similarities = pairwise_functional_similarity(scores)
    
    assert similarities[0, 1] < 0.5
    assert similarities[0, 2] < 0.5
    assert similarities[1, 2] < 0.5