import pytest 
import faiss
import numpy as np

from askme.utils import TextEmbeddingWithChunker
from askme.rtp.make_collection_index import make_faiss_index
from askme.rtp.label_propagation import make_knn_graph, sparse_affinity, propagate_labels

    
def test_make_faiss_cpu_index():
    texts = [
        "This is a sample document." * 10,
        "Another document goes here." * 10,
        "More text data for testing." * 10,
        "Dinossaurs are very fun animals to learn about.",
        "Dogs and cats are common pets.",
        "Computers have revolutionized the world.",
    ]
    embedding_model = TextEmbeddingWithChunker()
    
    dimension = 768  # Dimension for 'all-MiniLM-L6-v2'
    index = make_faiss_index(
        text_collection=texts,
        embedding_model=embedding_model,
        dimension=dimension,
        use_gpu=False,
    )
    assert index.ntotal == len(texts)
