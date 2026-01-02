import askme.rtp as rtp
from askme.rtp.make_collection_index import make_faiss_index
from askme.utils import TextEmbeddingWithChunker
import faiss
from tqdm import tqdm
import numpy as np

small_text_collection = [
    "The cat sat on the mat.",
    "The cat in in the box." ,
    "The dog barked loudly.",
    "I like cats",
    "I like dogs",
    "The dog is in the yard.",
    "Birds can fly high in the sky.",
    "Fish swim in the ocean.",
    "Elephants are the largest land animals.",
    "Lions are known as the kings of the jungle.",
]

def main():
    # Step one: vectorize documents
    print("Vectorizing documents...")
    model_name = 'sentence-transformers/all-MiniLM-L6-v2'
    model_name = 'sentence-transformers/paraphrase-albert-small-v2'
    embedding_model = TextEmbeddingWithChunker(
        model_name=model_name,
        chunk_size=50,
        overlap=10,
        device='cpu',
    )
    print("Making FAISS index...")
    faiss_index = make_faiss_index(
        text_collection=small_text_collection,
        embedding_model=embedding_model,
        dimension=embedding_model.model.get_sentence_embedding_dimension(),
        use_gpu=False,
    )
    
    print("Getting medoids...")
    # Step two: get 3 medoids via k-means
    n_clusters = 3
    kmeans = faiss.Kmeans(
        d=embedding_model.model.get_sentence_embedding_dimension(),
        k=n_clusters,
        niter=20,
        verbose=True,
        gpu=False,
    )
    kmeans.train(faiss_index.reconstruct_n(0, faiss_index.ntotal))
    # medoid indexes
    _, medoid_indexes = kmeans.index.search(
        faiss_index.reconstruct_n(0, faiss_index.ntotal),
        1,
    )
    
    medoids = []
    print("Medoids:")
    print(medoid_indexes)
    for medoid_index in medoid_indexes:
        medoids.append(small_text_collection[medoid_index[0]])
    print(medoids)
    
if __name__ == "__main__":
    main()