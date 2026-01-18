"""
Inference testing script for RTP trees.

This script iterates through a list of documents and runs the inference procedure
using the query function to classify documents into an RTP tree.

Usage:
    # Using HuggingFace dataset
    python inference.py --tree tree.pkl --source huggingface --dataset fancyzhx/ag_news --n_samples 100

    # Using local .txt files
    python inference.py --tree tree.pkl --source local --input_dir ./documents
"""

import os
import sys
import argparse
import pickle
import json
from pathlib import Path
from typing import List, Tuple
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from askme.rtp import query, TreeNode
from askme.askquestions import models


def load_tree(tree_path: str) -> TreeNode:
    """
    Load a pre-built RTP tree from a file.
    
    Args:
        tree_path: Path to the tree file (.pkl or .json)
        
    Returns:
        TreeNode: The loaded tree
    """
    print(f"Loading tree from {tree_path}...")
    
    if tree_path.endswith('.pkl'):
        with open(tree_path, 'rb') as f:
            tree = pickle.load(f)
    elif tree_path.endswith('.json'):
        with open(tree_path, 'r') as f:
            json_data = json.load(f)
            tree = TreeNode.model_validate(json_data)
    else:
        raise ValueError("Tree file must be .pkl or .json format")
    
    print(f"Tree loaded successfully")
    return tree


def load_documents_from_huggingface(
    dataset_name: str,
    split: str = 'test',
    n_samples: int = None,
    seed: int = 42
) -> Tuple[List[str], List[int]]:
    """
    Load documents from a HuggingFace dataset.
    
    Args:
        dataset_name: Name of the HuggingFace dataset
        split: Dataset split to use
        n_samples: Number of samples to load (None for all)
        seed: Random seed for sampling
        
    Returns:
        Tuple of (texts, labels)
    """
    from datasets import load_dataset
    
    print(f"Loading {n_samples if n_samples else 'all'} samples from {dataset_name} ({split} split)...")
    dataset = load_dataset(dataset_name, split=split)
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Sample indices
    total_size = len(dataset)
    if n_samples is None or n_samples >= total_size:
        indices = list(range(total_size))
    else:
        indices = np.random.choice(total_size, size=n_samples, replace=False)
    
    # Extract texts and labels
    texts = []
    labels = []
    skipped_count = 0
    for idx in tqdm(indices, desc="Loading documents"):
        item = dataset[int(idx)]
        # Handle different dataset formats
        if 'text' in item:
            text = item['text']
        elif 'content' in item:
            text = item['content']
        elif 'document' in item:
            text = item['document']
        else:
            # Try to combine title and text if available
            text = ' '.join(filter(None, [item.get('title', ''), item.get('description', '')]))
            text = text.strip()
            if not text:  # If still empty, skip this item
                skipped_count += 1
                continue
        
        texts.append(text)
        
        # Extract label if available
        if 'label' in item:
            labels.append(item['label'])
        else:
            labels.append(-1)  # No label available
    
    if skipped_count > 0:
        print(f"Warning: Skipped {skipped_count} empty document(s)")
    print(f"Loaded {len(texts)} documents")
    if any(label != -1 for label in labels):
        valid_labels = [l for l in labels if l != -1]
        if valid_labels:
            print(f"Label distribution: {np.bincount(valid_labels)}")
    
    return texts, labels


def load_documents_from_local(input_dir: str) -> Tuple[List[str], List[str]]:
    """
    Load documents from local .txt files.
    
    Args:
        input_dir: Directory containing .txt files
        
    Returns:
        Tuple of (texts, filenames)
    """
    print(f"Loading documents from {input_dir}...")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        raise ValueError(f"Input directory does not exist: {input_dir}")
    
    txt_files = sorted(input_path.glob("*.txt"))
    if not txt_files:
        raise ValueError(f"No .txt files found in {input_dir}")
    
    texts = []
    filenames = []
    skipped_files = []
    
    for txt_file in tqdm(txt_files, desc="Loading documents"):
        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                text = f.read().strip()
                if text:  # Skip empty files
                    texts.append(text)
                    filenames.append(txt_file.name)
                else:
                    skipped_files.append(txt_file.name)
        except (IOError, UnicodeDecodeError) as e:
            print(f"\nWarning: Could not read {txt_file.name}: {e}")
            skipped_files.append(txt_file.name)
    
    if skipped_files:
        print(f"Warning: Skipped {len(skipped_files)} empty or unreadable file(s)")
    print(f"Loaded {len(texts)} documents from {len(txt_files)} files")
    
    return texts, filenames


def run_inference(
    documents: List[str],
    tree: TreeNode,
    nli_model,
    nli_tokenizer,
    device: str = 'cpu',
    chunk_size: int = 200,
    overlap: int = 20
) -> List[TreeNode]:
    """
    Run inference on a list of documents using the query function.
    
    Args:
        documents: List of text documents to classify
        tree: The RTP tree to query
        nli_model: NLI model for entailment checking
        nli_tokenizer: Tokenizer for the NLI model
        device: Device to run on ('cpu' or 'cuda')
        chunk_size: Size of text chunks for NLI processing
        overlap: Overlap between chunks
        
    Returns:
        List of leaf TreeNodes, one for each document
    """
    print(f"\nRunning inference on {len(documents)} documents...")
    
    results = []
    for doc in tqdm(documents, desc="Running inference"):
        leaf = query(
            document=doc,
            tree_root=tree,
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            device=device,
            chunk_size=chunk_size,
            overlap=overlap
        )
        results.append(leaf)
    
    return results


def print_inference_results(
    documents: List[str],
    results: List[TreeNode],
    identifiers: List,
    max_doc_length: int = 100
):
    """
    Print inference results in a readable format.
    
    Args:
        documents: Original documents
        results: Leaf nodes from inference
        identifiers: Document identifiers (labels or filenames)
        max_doc_length: Maximum length of document text to display
    """
    print("\n" + "="*80)
    print("INFERENCE RESULTS")
    print("="*80)
    
    for i, (doc, leaf, identifier) in enumerate(zip(documents, results, identifiers)):
        # Truncate document for display
        doc_display = doc[:max_doc_length] + "..." if len(doc) > max_doc_length else doc
        
        print(f"\nDocument {i+1} (ID: {identifier}):")
        print(f"  Text: {doc_display}")
        print(f"  Classified to leaf with {len(leaf.documents)} documents")
        if leaf.question:
            print(f"  Leaf question: {leaf.question}")
        
        # Show sample documents from the leaf
        if leaf.documents and len(leaf.documents) <= 5:
            print(f"  Leaf document indices: {leaf.documents}")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    # Count unique leaves (using id() is safe here as we're only comparing within this session)
    unique_leaves = set(id(leaf) for leaf in results)
    print(f"Total documents processed: {len(documents)}")
    print(f"Unique leaf nodes reached: {len(unique_leaves)}")
    
    # Show distribution of documents per leaf
    leaf_counts = {}
    for leaf in results:
        leaf_id = id(leaf)  # Using id() to group documents by the same leaf instance
        leaf_counts[leaf_id] = leaf_counts.get(leaf_id, 0) + 1
    
    print(f"\nDocuments per leaf distribution:")
    for leaf_id, count in sorted(leaf_counts.items(), key=lambda x: -x[1]):
        print(f"  Leaf {leaf_id}: {count} document(s)")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference on documents using a pre-built RTP tree"
    )
    
    # Required arguments
    parser.add_argument(
        '--tree',
        type=str,
        required=True,
        help='Path to the pre-built tree file (.pkl or .json)'
    )
    
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        choices=['huggingface', 'local'],
        help='Source of documents: "huggingface" for HF datasets, "local" for .txt files'
    )
    
    # HuggingFace-specific arguments
    parser.add_argument(
        '--dataset',
        type=str,
        help='HuggingFace dataset name (required if source=huggingface)'
    )
    
    parser.add_argument(
        '--split',
        type=str,
        default='test',
        help='Dataset split to use (default: test)'
    )
    
    parser.add_argument(
        '--n_samples',
        type=int,
        default=None,
        help='Number of samples to load (default: all)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )
    
    # Local file arguments
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Directory containing .txt files (required if source=local)'
    )
    
    # Inference parameters
    parser.add_argument(
        '--device',
        type=str,
        default='cpu',
        choices=['cpu', 'cuda'],
        help='Device to run inference on (default: cpu)'
    )
    
    parser.add_argument(
        '--chunk_size',
        type=int,
        default=200,
        help='Size of text chunks for NLI processing (default: 200)'
    )
    
    parser.add_argument(
        '--overlap',
        type=int,
        default=20,
        help='Overlap between chunks (default: 20)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.source == 'huggingface' and not args.dataset:
        parser.error("--dataset is required when source=huggingface")
    
    if args.source == 'local' and not args.input_dir:
        parser.error("--input_dir is required when source=local")
    
    # Load the tree
    tree = load_tree(args.tree)
    
    # Load documents
    if args.source == 'huggingface':
        documents, identifiers = load_documents_from_huggingface(
            dataset_name=args.dataset,
            split=args.split,
            n_samples=args.n_samples,
            seed=args.seed
        )
    else:  # local
        documents, identifiers = load_documents_from_local(args.input_dir)
    
    # Initialize NLI model
    print(f"\nInitializing NLI model on {args.device}...")
    nli_model, nli_tokenizer = models.make_nli_model(device=args.device)
    print("NLI model loaded successfully")
    
    # Run inference
    results = run_inference(
        documents=documents,
        tree=tree,
        nli_model=nli_model,
        nli_tokenizer=nli_tokenizer,
        device=args.device,
        chunk_size=args.chunk_size,
        overlap=args.overlap
    )
    
    # Print results
    print_inference_results(documents, results, identifiers)


if __name__ == "__main__":
    main()
