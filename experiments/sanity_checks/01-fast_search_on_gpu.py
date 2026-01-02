from askme.utils import TextEmbeddingWithChunker
from askme.rtp.make_collection_index import make_faiss_index
import time 

texts = [
        "This is a sample document." * 10,
        "Another document goes here." * 10,
        "More text data for testing." * 10,
        "Dinossaurs are very fun animals to learn about.",
        "Dogs and cats are common pets.",
        "Computers have revolutionized the world.",
    ]
t0 = time.time()
embedding_model = TextEmbeddingWithChunker(
    model_name='all-MiniLM-L6-v2',
    chunk_size=10,
    overlap=2,
    device='cuda',
)
t1 = time.time()
print(f"Time to load model: {t1 - t0:.2f} seconds")

dimension = 384  # Dimension for 'all-MiniLM-L6-v2'

t0 = time.time()
index = make_faiss_index(
    text_collection=texts,
    embedding_model=embedding_model,
    dimension=dimension,
)
t1 = time.time()
print(f"Time to create FAISS GPU index: {t1 - t0:.2f} seconds")

t0 = time.time()
query = "tell me about pets"
query_embedding = embedding_model(query)
query_embedding = query_embedding.astype('float32').reshape(1, -1)
t1 = time.time()
print(f"Time to embed query: {t1 - t0:.2f} seconds")
     
t0 = time.time()
k = 2
distances, indices = index.search(query_embedding, k)
t1 = time.time()
print(f"Time to search index: {t1 - t0:.2f} seconds")
print("Top-k results:")
for i, idx in enumerate(indices[0]):
    print(f"Rank {i+1}: Document Index {idx}, Distance: {distances[0][i]}")
    print(f"Document: {texts[idx]}")

