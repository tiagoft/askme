import faiss
from ..utils import TextEmbeddingWithChunker
from collections.abc import Iterable


def make_faiss_gpu_index(
    text_collection: Iterable[str],
    embedding_model: TextEmbeddingWithChunker,
    dimension: int,
) -> faiss.Index:
    res = faiss.StandardGpuResources()

    # Create a GPU index directly
    gpu_index = faiss.IndexFlatL2(dimension)
    gpu_index = faiss.index_cpu_to_gpu(res, 0, gpu_index)

    for text in text_collection:
        embedding = embedding_model(text)
        embedding = embedding.astype('float32').reshape(1, -1)
        gpu_index.add(embedding)
    
    return gpu_index



