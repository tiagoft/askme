import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from askme.rtp import (
    RTPBuilder,
    RTPRecursion,
    run_hdbscan_baseline,
    evaluate_exploratory_power,
    calculate_tree_depth,
)

from datasets import load_dataset
import numpy as np
from typing import List, Tuple
from tqdm import tqdm
from askme.askquestions import check_entailment, models
from askme.utils import chunk_text
from transformers import AutoModelForSequenceClassification, AutoTokenizer

def load_agnews_sample(n_samples: int = 500, seed: int = 42) -> Tuple[List[str], List[int]]:
    """
    Load a sample of n documents from the AG News dataset.
    
    Args:
        n_samples: Number of samples to load
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (texts, labels)
    """
    print(f"Loading {n_samples} samples from AG News dataset...")
    dataset = load_dataset('fancyzhx/ag_news', split='test')
    
    # Set random seed for reproducibility
    np.random.seed(seed)
    
    # Sample n_samples indices
    total_size = len(dataset)
    indices = np.random.choice(total_size, size=min(n_samples, total_size), replace=False)
    
    # Extract texts and labels
    texts = []
    labels = []
    for idx in indices:
        item = dataset[int(idx)]
        # Combine title and description for richer text
        text = item['text']
        texts.append(text)
        labels.append(item['label'])
    
    print(f"Loaded {len(texts)} documents")
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Classes: 0=World, 1=Sports, 2=Business, 3=Sci/Tech")
    
    return texts, labels

def main():
    small_text_collection, labels = load_agnews_sample(n_samples=25, seed=42)
    answers = -np.ones((len(small_text_collection),), dtype=object)
    nli_model_name = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
    model, tokenizer = models.make_nli_model(model_name=nli_model_name)
    hypothesis = "This text is about sports."
    for doc_index in tqdm(range(len(small_text_collection))):
        document = small_text_collection[doc_index]
        chunked_text = chunk_text(document, chunk_size=200, overlap=20)
        pooled_results = check_entailment.pool_nli_scores(
            check_fn=check_entailment.check_entailment_nli,
            premise=document,
            hypothesis=hypothesis,
            chunk_size=200,
            overlap=20,
            model=model,
            tokenizer=tokenizer,
            device='cpu',
        )
        entails, entailment_score, contradiction_score, P_entailment = pooled_results
        if entails:
            answers[doc_index] = 1
        else:
            answers[doc_index] = 0
            
    actual_sports = [1 if labels[doc_index] == 1 else 0 for doc_index in range(len(labels))]
    correct = sum([1 if answers[i] == actual_sports[i] else 0 for i in range(len(answers))])
    total = len(answers)
    print(f"Correctly identified {correct} out of {total} documents as sports-related.")
    print(f"Accuracy: {correct / total:.2%}")
    print("Answers:", answers)
    print("Actual sports labels:", actual_sports)

if __name__ == "__main__":
    main()