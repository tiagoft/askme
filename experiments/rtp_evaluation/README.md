# RTP Evaluation Experiments

This directory contains experiments for evaluating the RTP (Recursive Tree Partitioning) method against baseline approaches.

## test_small_agnews.py

A comprehensive evaluation script that compares RTP Recursion against HDBSCAN baseline using the AG News dataset.

### Overview

The script performs the following tasks:

1. **Data Loading**: Loads a sample of 100 documents from the `fancyzhx/ag_news` dataset (test split)
   - AG News contains news articles in 4 categories: World, Sports, Business, Sci/Tech
   - Uses random sampling with a fixed seed for reproducibility

2. **RTP Evaluation**: 
   - Builds a recursive RTP tree using LLM-generated questions
   - Reports tree structure, split metrics, and computational costs
   - Evaluates exploratory power (purity, entropy, isolation depth)

3. **HDBSCAN Evaluation**:
   - Builds a hierarchical tree using density-based clustering
   - Uses sentence embeddings for similarity computation
   - Evaluates exploratory power with the same metrics

4. **Comparison**:
   - Compares both methods on purity, entropy, tree depth, and number of leaves
   - Highlights the trade-offs between interpretability (RTP) and algorithmic efficiency (HDBSCAN)

### Requirements

The script requires the following dependencies (specified in `pyproject.toml`):
- `datasets` (HuggingFace datasets library)
- `sentence-transformers`
- `faiss-gpu-cu12` or `faiss-cpu`
- `scikit-learn`
- `numpy`
- `pydantic-ai` (for LLM calls)
- `transformers` (for NLI models)

### Usage

```bash
# From the repository root
python experiments/rtp_evaluation/test_small_agnews.py
```

### Expected Output

The script produces:
- Tree structures with sample documents for both methods
- Detailed metrics including:
  - Tree depth and number of leaves
  - Average leaf purity and entropy
  - Isolation depths by class
  - RTP-specific metrics (LLM token usage, NLI calls, timing)
- A comparison summary highlighting strengths of each approach

### Configuration

The script uses the following default parameters:

**RTP Parameters:**
- `n_medoids=3`: Number of medoids for k-means clustering
- `n_documents_to_answer=10`: Documents to label via NLI
- `min_node_size=5`: Minimum documents in a node before splitting
- `min_split_ratio=0.1`: Minimum proportion in smaller child
- `max_split_ratio=0.9`: Maximum proportion in larger child
- `max_depth=4`: Maximum tree depth

**HDBSCAN Parameters:**
- `model_name="all-MiniLM-L6-v2"`: Sentence transformer model
- `min_cluster_size=5`: Minimum points in a cluster
- `min_samples=2`: Minimum samples for core points
- `device="cpu"`: Device for computation

**Dataset Parameters:**
- `n_samples=100`: Number of documents to sample
- `seed=42`: Random seed for reproducibility

These can be modified in the script as needed for different experiments.

### Notes

- The script requires API access for LLM calls (configured via pydantic-ai)
- Execution time depends on the LLM API response times and NLI model inference
- GPU acceleration is available but not required (set `use_gpu=True` for RTPBuilder and `device="cuda"` for HDBSCAN)
