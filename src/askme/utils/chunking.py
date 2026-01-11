from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
import numpy as np
import pickle

from torch.utils.data import Dataset, DataLoader
import torch


class ChunkedBatchedModel:

    def __init__(
        self,
        model,
        chunk_size=512,
        overlap=128,
        batch_size=8,
        pooling_strategy='mean',
    ):
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.batch_size = batch_size
        self.model = model
        if pooling_strategy not in ['mean', 'max']:
            raise ValueError(
                f"Unsupported pooling strategy: {pooling_strategy}")
        self.pooling_strategy = pooling_strategy

    def __call__(self, X: list[str]):
        dataset = ChunkingDataset(
            X,
            chunk_fn=lambda item: chunk_text(
                item,
                self.chunk_size,
                self.overlap,
            ),
        )
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=smart_collate,
        )

        embeddings_per_item = []
        for batch in dataloader:
            chunk_embeddings = self.model(batch['chunks'])

            if self.pooling_strategy == 'mean':
                pool_fn = lambda embs: embs.mean(dim=0)
            elif self.pooling_strategy == 'max':
                pool_fn = lambda embs: embs.max(dim=0).values

            for start, end in batch['boundaries']:
                item_embs = chunk_embeddings[start:end]
                pooled = pool_fn(item_embs)
                embeddings_per_item.append(pooled)

        return torch.stack(embeddings_per_item)


class ChunkingDataset(Dataset):

    def __init__(self, data, chunk_fn):
        self.data = data
        self.chunk_fn = chunk_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunks = self.chunk_fn(self.data[idx])
        # We return index + chunks → easy to regroup later
        return {
            'item_idx': idx,
            'chunks': chunks,  # list of chunks
            'num_chunks': len(chunks)
        }


def smart_collate(batch):
    """Flattens chunks but keeps track of boundaries"""
    item_indices = []
    all_chunks = []
    chunk_boundaries = []  # where each original item starts and ends

    current_pos = 0
    for sample in batch:
        item_idx = sample['item_idx']
        chunks = sample['chunks']

        item_indices.extend([item_idx] * len(chunks))
        all_chunks.extend(chunks)
        chunk_boundaries.append((current_pos, current_pos + len(chunks)))
        current_pos += len(chunks)

    return {
        'chunks': all_chunks,  # list of chunks (what model will get)
        'item_indices': item_indices,  # [item_idx for each chunk]
        'boundaries':
        chunk_boundaries,  # list of (start, end) for each original item
        'original_batch_size': len(batch)
    }


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


class TextEmbeddingWithChunker:

    def __init__(self,
                 model_name: str,
                 chunk_size: int = 350,
                 overlap: int = 50,
                 pooling_fn: callable = np.mean,
                 device: str = 'cpu'):

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.pooling_fn = pooling_fn
        self.cache = {}

    def __call__(self, text: str) -> np.ndarray:
        if text in self.cache:
            return self.cache[text]
        chunks = chunk_text(text, self.chunk_size, self.overlap)
        chunk_embeddings = self.model.encode(chunks)
        embedding = self.pooling_fn(chunk_embeddings, axis=0)
        self.cache[text] = embedding
        return embedding

    def save_cache(self, path: str, append: bool = False):
        if append:
            try:
                with open(path, 'rb') as f:
                    current_cache = pickle.load(f)
                self.cache.update(current_cache)
            except FileNotFoundError:
                pass
        with open(path, 'wb') as f:
            pickle.dump(self.cache, f)

    def load_cache(self, path: str):
        try:
            with open(path, 'rb') as f:
                self.cache = pickle.load(f)
        except FileNotFoundError:
            return {}


class ChunkingDataset(Dataset):

    def __init__(self, data, chunk_fn):
        self.data = data
        self.chunk_fn = chunk_fn

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        chunks = self.chunk_fn(self.data[idx])
        # We return index + chunks → easy to regroup later
        return {
            'item_idx': idx,
            'chunks': chunks,  # list of chunks
            'num_chunks': len(chunks)
        }


def chunked_collate(batch):
    """Flattens chunks but keeps track of boundaries"""
    item_indices = []
    all_chunks = []
    chunk_boundaries = []  # where each original item starts and ends

    current_pos = 0
    for sample in batch:
        item_idx = sample['item_idx']
        chunks = sample['chunks']

        item_indices.extend([item_idx] * len(chunks))
        all_chunks.extend(chunks)
        chunk_boundaries.append((current_pos, current_pos + len(chunks)))
        current_pos += len(chunks)

    return {
        'chunks': all_chunks,  # list of chunks (what model will get)
        'item_indices': item_indices,  # [item_idx for each chunk]
        'boundaries':
        chunk_boundaries,  # list of (start, end) for each original item
        'original_batch_size': len(batch)
    }
