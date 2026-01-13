from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle
from tqdm import tqdm 

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


def _pool(logits, boundaries, label_idx):
    """Pool logits based on boundaries, selecting max per original item."""
    pooled = []
    for start, end in boundaries:
        item_logits = logits[start:end, label_idx]
        max_logit = torch.max(item_logits).item()
        pooled.append(max_logit)
    return pooled
class NLIWithChunkingAndPooling:

    def __init__(self,
                 nli_model,
                 tokenizer,
                 batch_size: int = 8,
                 chunk_size: int = 350,
                 overlap: int = 50,
                 device: str = 'cuda:0',
                 chunk_fn: callable = chunk_text,
                 label_names=["entailment", "neutral", "contradiction"]):
        self.nli_model = nli_model
        self.tokenizer = tokenizer
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.device = device
        self.label_names = label_names
        self.chunk_fn = chunk_fn
        self.batch_size = batch_size
        
    def __call__(
        self,
        premise: list[str],
        hypothesis: str,
        **kwargs,
    ) -> list[tuple[bool, float, float, float]]:
        """
        Check entailment between premise and hypothesis using chunking and max-pooling.
        
        Args:
            premise: The premise text
            hypothesis: The hypothesis text
            **kwargs: Additional arguments to pass to the NLI model
        Returns:
            Tuple of (is_entailed, entailment_score, contradiction_score, P_entailment)
            from the chunk with the highest entailment score (max-pooling)
        """
        dataset = ChunkingDataset(
            data=premise,
            chunk_fn=lambda text: self.chunk_fn(
                text,
                chunk_size=self.chunk_size,
                overlap=self.overlap,
            )
        )
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=chunked_collate,
        )
        
        all_results = []
        for data in tqdm(loader):
            # Prepare inputs for NLI model
            inputs = self.tokenizer(
                data['chunks'],
                [hypothesis] * len(data['chunks']),
                return_tensors='pt',
                padding=True,
                truncation=True,
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.nli_model(**inputs, **kwargs)
                logits = outputs.logits
            
            boundaries = data['boundaries']
            
            entailment_scores = _pool(
                logits,
                boundaries,
                label_idx=self.label_names.index("entailment"),
            )
            
            contradiction_scores = _pool(
                logits,
                boundaries,
                label_idx=self.label_names.index("contradiction"),
            )
            
            P_entailment = [
                torch.softmax(
                    torch.tensor([e, c]), dim=0
                )[0].item()
                for e, c in zip(entailment_scores, contradiction_scores)
            ]
            
            results = []
            for e_score, c_score, p_e in zip(
                entailment_scores,
                contradiction_scores,
                P_entailment
            ):
                is_entailed = e_score > c_score
                results.append((is_entailed, e_score, c_score, p_e))
            
            all_results.extend(results)

        return all_results
                