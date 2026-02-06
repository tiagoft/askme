from sentence_transformers import SentenceTransformer
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from askme.config.config import NLIBatchingChukingConfig, config_factory
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np
import pickle
from tqdm import tqdm
import re
from pydantic import BaseModel


class NLIResults(BaseModel):
    hypothesis: str
    is_entailed: bool
    entailment_score: float
    contradiction_score: float
    neutral_score: float
    P_entailment_binary: float
    P_contradiction_binary: float
    P_entailment_ternary: float
    P_contradiction_ternary: float
    P_neutral_ternary: float
    entropy_binary: float
    entropy_ternary: float
    config: NLIBatchingChukingConfig | None = None


def chunk_text(
    text: str,
    chunk_size: int = 350,
    overlap: int = 50,
    max_characters: int = 1000,
) -> list[str]:
    """
    Split text into chunks of specified word count with overlap.
    
    Args:
        text: The text to split into chunks
        chunk_size: Number of words per chunk (default: 350)
        overlap: Number of words to overlap between chunks (default: 50)
        max_characters: Maximum number of characters to use in each chunk (default: 1000)
    
    Returns:
        List of text chunks
    """
    ascii_text = text.encode('ascii', 'ignore').decode('ascii')
    ascii_text = ''.join(c for c in ascii_text
                         if c.isprintable() or c in '\n\t ')
    ascii_text = ' '.join(ascii_text.split())
    ascii_text = re.sub(r'\S{15,}', '', ascii_text)
    ascii_text = re.sub(r'--+', '', ascii_text)
    ascii_text = re.sub(r'cut\s+here[.]+end\s+cut\s+here', '', ascii_text)

    words = ascii_text.split()
    words = [
        word for word in words
        if re.fullmatch(r'[a-zA-Z0-9]+', word) and len(word) < 10
    ]
    if len(words) <= chunk_size:
        if len(ascii_text) > max_characters:
            return [ascii_text[:max_characters]]
        return [ascii_text]

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = ' '.join(words[start:end])
        chunk = chunk[:max_characters]
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

    def __init__(
        self,
        chunk_fn: callable = chunk_text,
        disable_tqdm: bool = False,
        config: NLIBatchingChukingConfig = config_factory(
            NLIBatchingChukingConfig),
    ):
        self.config = config
        self.model_name = config.model_name
        self.chunk_size = config.chunk_size
        self.overlap = config.overlap
        if config.device == 'auto':
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = config.device

        self.label_names = config.label_names
        self.chunk_fn = chunk_fn
        self.batch_size = config.batch_size
        self.max_chunks_per_minibatch = config.max_chunks_per_minibatch
        self.max_characters_per_chunk = config.max_characters_per_chunk
        self.disable_tqdm = disable_tqdm

        self.nli_model = AutoModelForSequenceClassification.from_pretrained(
            config.model_name).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name,
                                                       use_fast=True)

    def __call__(
        self,
        premise: list[str],
        hypothesis: str,
        **kwargs,
    ) -> list[NLIResults]:
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

        assert isinstance(premise, list), "Premise must be a list of strings."
        dataset = ChunkingDataset(
            data=premise,
            chunk_fn=lambda text: self.chunk_fn(text,
                                                chunk_size=self.chunk_size,
                                                overlap=self.overlap,
                                                max_characters=self.
                                                max_characters_per_chunk))
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            collate_fn=chunked_collate,
        )

        all_results = []
        for data in tqdm(loader, disable=self.disable_tqdm):
            # Prepare inputs for NLI model
            try:
                inputs = self.tokenizer(
                    data['chunks'],
                    [hypothesis] * len(data['chunks']),
                    return_tensors='pt',
                    padding=True,
                    truncation="only_first",
                    max_length=512,
                    return_overflowing_tokens=False,
                ).to(self.device)
            except ValueError as e:
                print(f"Error tokenizing chunks: {e}")
                print(data['chunks'])
                print(hypothesis)

            model_inputs = {
                k: v
                for k, v in inputs.items() if k not in [
                    'overflowing_tokens', 'num_truncated_tokens',
                    'overflow_to_sample_mapping'
                ]
            }
            inputs = model_inputs

            logits_list = []
            for start_idx in range(0, len(data['chunks']),
                                   self.max_chunks_per_minibatch):
                end_idx = start_idx + self.max_chunks_per_minibatch
                minibatch_inputs = {
                    k: v[start_idx:end_idx]
                    for k, v in inputs.items()
                }
                with torch.no_grad():
                    outputs = self.nli_model(**minibatch_inputs, **kwargs)
                    logits_ = outputs.logits
                    logits_list.append(logits_)
                del minibatch_inputs  # free memory
                torch.cuda.empty_cache()
            logits = torch.cat(logits_list, dim=0)

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

            neutral_scores = _pool(
                logits,
                boundaries,
                label_idx=self.label_names.index("neutral"),
            )

            P_binary = [
                torch.softmax(torch.tensor([e, c]), dim=0) for e, c in zip(
                    entailment_scores,
                    contradiction_scores,
                )
            ]
            P_ternary = [
                torch.softmax(torch.tensor([e, c, n]), dim=0)
                for e, c, n in zip(
                    entailment_scores,
                    contradiction_scores,
                    neutral_scores,
                )
            ]

            for pbin, pter in zip(P_binary, P_ternary):
                is_entailed = pbin[0].item() > pbin[1].item()
                result = NLIResults(
                    hypothesis=hypothesis,
                    is_entailed=is_entailed,
                    entailment_score=pter[0].item(),
                    contradiction_score=pter[1].item(),
                    neutral_score=pter[2].item(),
                    P_entailment_binary=pbin[0].item(),
                    P_contradiction_binary=pbin[1].item(),
                    P_entailment_ternary=pter[0].item(),
                    P_contradiction_ternary=pter[1].item(),
                    P_neutral_ternary=pter[2].item(),
                    entropy_binary=-torch.sum(pbin * torch.log2(pbin + 1e-10)).item(),
                    entropy_ternary=-torch.sum(pter * torch.log2(pter + 1e-10)).item(),
                    config = self.config,
                )
                all_results.append(result)

        return all_results
