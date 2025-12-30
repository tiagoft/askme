from sentence_transformers import SentenceTransformer
import numpy as np 

def chunk_text(
    text: str,
    chunk_size: int = 350,
    overlap: int = 50,
) -> list[str]:
    """
    Split text into chunks of specified word count with overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: Number of words per chunk (default: 350)
        overlap: Number of words to overlap between chunks (default: 50)
    
    Returns:
        List of text chunks
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunks.append(chunk)

        # Move forward by (chunk_size - overlap) words
        start += (chunk_size - overlap)

        # Break if we've covered all words
        if end >= len(words):
            break

    return chunks


def embed_text(
    text: str,
    model_name: str = 'all-MiniLM-L6-v2',
    chunk_size: int = 350,
    overlap: int = 50,
    pooling_fn : callable = np.mean,
) -> np.ndarray:
    """
    Generate embedding for the given text using a SentenceTransformer model.
    
    Args:
        text: The text to embed
        model_name: The name of the SentenceTransformer model to use (default: 'all-MiniLM-L6-v2')
        chunk_size: Number of words per chunk for embedding (default: 350)
        overlap: Number of words to overlap between chunks (default: 50)
    
    """
    model = SentenceTransformer(model_name)
    chunks = chunk_text(text, chunk_size, overlap)
    chunk_embeddings = model.encode(chunks)

    # Average the embeddings of all chunks to get a single embedding
    embedding = pooling_fn(chunk_embeddings, axis=0)
    return embedding