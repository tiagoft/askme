"""Tests for RTPBuilder class."""

import pytest
from askme.rtp import RTPBuilder, TreeNode, SplitMetrics
from pprint import pprint

# Sample text collection for testing
sample_text_collection = [
    "The cat sat on the mat.",
    "The cat is in the box.",
    "The dog barked loudly.",
    "I like cats",
    "I like dogs",
    "The dog is in the yard.",
    "Birds can fly high in the sky.",
    "Fish swim in the ocean.",
    "Elephants are the largest land animals.",
    "Lions are known as the kings of the jungle.",
]


def main():
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    result, metrics = builder(sample_text_collection, return_metrics=True)
    
    pprint(metrics)
    pprint(result)
    
if __name__ == "__main__":
    main()