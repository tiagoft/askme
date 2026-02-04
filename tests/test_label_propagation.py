import pytest 
import faiss
import numpy as np

from askme.utils import TextEmbeddingWithChunker
from askme.rtp.make_collection_index import make_faiss_index
from askme.rtp.label_propagation import make_knn_graph, sparse_affinity, propagate_labels, LabelPropagation
from askme.config.config import LabelPropagationConfig, config_factory

def test_label_propagation():
    # Create a simple FAISS index with known embeddings
    
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],  # Point 0
        [0.9, 0.1, 0.0, 0.0],  # Point 1
        [0.0, 1.0, 0.0, 0.0],  # Point 2
        [0.0, 0.9, 0.1, 0.0],  # Point 3
        [0.0, 0.0, 0.0, 1.0],  # Point 4
    ]
    
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    
    indices, distances = make_knn_graph(np.array(embeddings).astype('float32'), index, n_neighbors=2)
    W = sparse_affinity(indices, distances, sigma=1.0)
    
    # Labels: -1 for unlabeled, class index otherwise
    y = np.array([0, -1, 1, -1, -1])
    
    F = propagate_labels(W, y, alpha=0.01, max_iter=100, tol=1e-3)
    
    # Check that labels have been propagated correctly
    predicted_labels = F
    assert predicted_labels[1] == 0  # Point 1 should get label of Point 0
    assert predicted_labels[3] == 1  # Point 3 should get label of Point 2
    
def test_label_propagation_object():
    dimension = 4
    index = faiss.IndexFlatIP(dimension)
    embeddings = [
        [1.0, 0.0, 0.0, 0.0],  # Point 0
        [0.9, 0.1, 0.0, 0.0],  # Point 1
        [0.0, 1.0, 0.0, 0.0],  # Point 2
        [0.0, 0.9, 0.1, 0.0],  # Point 3
        [0.0, 0.0, 0.0, 1.0],  # Point 4
    ]
    
    embeddings = np.array(embeddings).astype('float32')
    faiss.normalize_L2(embeddings)
    index.add(embeddings)
    lp_config = LabelPropagationConfig(
        n_neighbors=2,
        sigma=1.0,
        alpha=0.01,
        max_iter=100,
        tol=1e-3,
    )
    lp = LabelPropagation(index, config=lp_config)
    y = np.array([0, -1, 1, -1, -1])
    predicted_labels = lp.fit_predict(embeddings, y)
    assert predicted_labels[1] == 0  # Point 1 should get label of Point 0
    assert predicted_labels[3] == 1  # Point 3 should get label of