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
            raise ValueError(f"Unsupported pooling strategy: {pooling_strategy}")
        self.pooling_strategy = pooling_strategy

    def __call__(self, X: list[str]):
        dataset = ChunkingDataset(
            X,
            chunk_fn=lambda item: chunk_fn(item, self.chunk_size, self.overlap
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
            
def chunk_fn(item, chunk_size=512, overlap=128):
    chunks = []
    start = 0
    while start < len(item):
        end = min(start + chunk_size, len(item))
        chunks.append(item[start:end])
        if end == len(item):
            break
        start += chunk_size - overlap
    return chunks


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


# ────────────────────────────────────────────────────────────────
# Example usage & post-processing
# ────────────────────────────────────────────────────────────────
def test():
    data = [
        torch.arange(10),
        torch.arange(20, 40),
    ]
    chunking_function = lambda x: chunk_fn(
        x,
        chunk_size=4,
        overlap=2,
    )
    dataset = ChunkingDataset(
        data,
        chunking_function,
    )
    loader = DataLoader(
        dataset,
        batch_size=3,
        shuffle=False,
        collate_fn=smart_collate,
    )

    for batch in loader:
        print(batch)
        break

        # ──── model forward ───────────────────────────────
        # chunk_embeddings = model(batch['chunks'])   # shape: [total_chunks, hidden_size]

        # Example: suppose we have embeddings for each chunk
        # chunk_embeddings = torch.randn(17, 768)       # fake

        # ──── regroup & pool ─────────────────────────────────
        # embeddings_per_item = []

        # for start, end in batch['boundaries']:
        #     item_embs = chunk_embeddings[start:end]      # all chunks of this item
        #     # Pick your favorite pooling strategy:
        #     pooled = item_embs.mean(dim=0)               # ← very common
        #     # pooled = item_embs.max(dim=0).values
        #     # pooled = item_embs[0]                      # first chunk
        #     # pooled = attention_pooling(item_embs)      # your custom impl

        #     embeddings_per_item.append(pooled)

        # # Now embeddings_per_item has one vector per original item
        # # → ready for classification, contrastive loss, etc.
        # break


if __name__ == "__main__":
    test()
