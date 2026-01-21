from typing import List, Tuple
from datasets import load_dataset
import numpy as np

def load_dataset_sample(
        n_samples: int | None = 500,
        seed: int = 42,
        dataset_name: str = 'fancyzhx/ag_news',
        split: str = 'train') -> Tuple[List[str], List[int]]:
    """
    Load a sample of n documents from the specified dataset.
    
    Args:
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (texts, labels)
    """
    #print(f"Loading {n_samples} samples from {dataset_name} dataset...")
    text_key = 'text'
    label_key = 'label'
    if dataset_name == 'wikipedia':
        # Load /mnt/data2/nlp_datasets/wikipedia.dump.jsonl
        dataset = load_dataset('json',
                               data_files=[f'/mnt/data3/nlp_datasets/wikipedia/{split}.metadata.jsonl'],
                               split='train')
        label_key = 'supercategory'
        print(f"Dataset keys: {dataset[0].keys()}")
    elif dataset_name == 'bills':
        # Load /mnt/data2/nlp_datasets/wikipedia.dump.jsonl
        dataset = load_dataset('json',
                               data_files=[f'/mnt/data3/nlp_datasets/bills/{split}.metadata.jsonl'],
                               split='train')
        print(dataset[0].keys())
        text_key = 'summary'
        label_key = 'topic'
    else:    
        dataset = load_dataset(dataset_name, split=split, cache_dir='/mnt/data2/tiago/.cache/huggingface/datasets')
    
    
    # Set random seed for reproducibility
    np.random.seed(seed)

    # Sample n_samples indices
    total_size = len(dataset)
    if n_samples is None:
        n_samples = total_size
    indices = np.random.choice(total_size,
                               size=min(n_samples, total_size),
                               replace=False)

    # Extract texts and labels
    texts = []
    labels = []
    for idx in indices:
        item = dataset[int(idx)]
        # Combine title and description for richer text
        text = item[text_key]
        texts.append(text)
        labels.append(item[label_key])
    print(f"Dataset has {len(set(labels))} unique labels.")
    #print(f"Loaded {len(texts)} documents")
    #print(f"Label distribution: {np.bincount(labels)}")
    #print(f"Classes: 0=World, 1=Sports, 2=Business, 3=Sci/Tech")

    return texts, labels

if __name__ == "__main__":
    # Test the function
    texts, labels = load_dataset_sample(n_samples=None, dataset_name='SetFit/20_newsgroups', split='train')
    print(f"Dataset has {len(texts)} samples.")
    print(f"Unique categories: {len(set(labels))}")
    print(f"Labels: {set(labels)}")