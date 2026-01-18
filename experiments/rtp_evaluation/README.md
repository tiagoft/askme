# RTP Evaluation Experiments

This directory contains experiments for evaluating the RTP (Recursive Tree Partitioning) method against baseline approaches.

## test_small_agnews.py

A comprehensive evaluation script that compares RTP Recursion against HDBSCAN and BERTopic baselines using the AG News dataset.

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

4. **BERTopic Evaluation**:
   - Builds a hierarchical tree using topic modeling
   - Creates topics with interpretable word representations
   - Evaluates exploratory power with the same metrics

5. **Comparison**:
   - Compares all three methods on purity, entropy, tree depth, and number of leaves
   - Highlights the trade-offs between interpretability and efficiency

### Requirements

The script requires the following dependencies (specified in `pyproject.toml`):
- `datasets>=2.0.0` (HuggingFace datasets library for loading AG News)
- `sentence-transformers>=5.2.0` (for text embeddings)
- `faiss-gpu-cu12>=1.13.1` (for efficient similarity search)
- `scikit-learn>=1.3.0` (includes HDBSCAN clustering)
- `bertopic>=0.16.0` (for topic modeling)
- `pydantic-ai>=1.25.1` (for LLM calls)
- `transformers>=4.57.3` (for NLI models)
- `torch>=2.9.1` (deep learning framework)
- `numpy` (transitive dependency, used for array operations)

### Usage

```bash
# From the repository root
python experiments/rtp_evaluation/test_small_agnews.py
```

### Expected Output

The script produces:
- Tree structures with sample documents for all three methods
- Detailed metrics including:
  - Tree depth and number of leaves
  - Average leaf purity and entropy
  - Isolation depths by class
  - RTP-specific metrics (LLM token usage, NLI calls, timing)
  - BERTopic-specific metrics (number of topics found)
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

**BERTopic Parameters:**
- `model_name="all-MiniLM-L6-v2"`: Sentence transformer model
- `nr_topics="auto"`: Number of topics (automatically determined)
- `device="cpu"`: Device for computation
- `calculate_probabilities=False`: Whether to calculate topic probabilities

**Dataset Parameters:**
- `n_samples=100`: Number of documents to sample
- `seed=42`: Random seed for reproducibility

These can be modified in the script as needed for different experiments.

### Notes

- The script requires API access for LLM calls (configured via pydantic-ai)
- Execution time depends on the LLM API response times and NLI model inference
- GPU acceleration is available but not required (set `use_gpu=True` for RTPBuilder and `device="cuda"` for HDBSCAN)

## inference.py

A script for running inference on documents using a pre-built RTP tree. This script iterates through a list of documents and classifies each one by traversing the tree using Natural Language Inference (NLI).

### Overview

The script provides functionality to:

1. **Load a pre-built RTP tree**: Supports both `.pkl` (pickle) and `.json` formats
2. **Load documents from multiple sources**:
   - HuggingFace datasets (any text dataset)
   - Local `.txt` files from a directory
3. **Run inference**: Uses the `query` function to classify each document into a leaf node
4. **Display results**: Shows which leaf each document was classified to and summary statistics

### Usage

#### Using HuggingFace Datasets

```bash
# Run inference on 100 samples from AG News
python experiments/rtp_evaluation/inference.py \
    --tree path/to/tree.pkl \
    --source huggingface \
    --dataset fancyzhx/ag_news \
    --split test \
    --n_samples 100

# Run on all documents in a dataset
python experiments/rtp_evaluation/inference.py \
    --tree path/to/tree.pkl \
    --source huggingface \
    --dataset fancyzhx/ag_news
```

#### Using Local Text Files

```bash
# Run inference on .txt files in a directory
python experiments/rtp_evaluation/inference.py \
    --tree path/to/tree.pkl \
    --source local \
    --input_dir ./documents
```

### Arguments

**Required Arguments:**
- `--tree`: Path to the pre-built tree file (`.pkl` or `.json`)
- `--source`: Source of documents (`huggingface` or `local`)

**HuggingFace-specific Arguments:**
- `--dataset`: HuggingFace dataset name (required if `source=huggingface`)
- `--split`: Dataset split to use (default: `test`)
- `--n_samples`: Number of samples to load (default: all documents)
- `--seed`: Random seed for sampling (default: `42`)

**Local File Arguments:**
- `--input_dir`: Directory containing `.txt` files (required if `source=local`)

**Inference Parameters:**
- `--device`: Device to run inference on (`cpu` or `cuda`, default: `cpu`)
- `--chunk_size`: Size of text chunks for NLI processing in words (default: `200`)
- `--overlap`: Overlap between chunks in words (default: `20`)

### Output

The script outputs:

1. **Per-document results**: Shows each document's text (truncated), the leaf it was classified to, and the number of documents in that leaf
2. **Summary statistics**: 
   - Total documents processed
   - Number of unique leaf nodes reached
   - Distribution of documents across leaves

### Example Output

```
INFERENCE RESULTS
================================================================================

Document 1 (ID: 0):
  Text: Wall St. Bears Claw Back Into the Black (Reuters) Reuters - Short-sellers, Wall...
  Classified to leaf with 23 documents
  Leaf question: Is this about business or finance?

Document 2 (ID: 1):
  Text: Carlyle Looks Toward Commercial Aerospace (Reuters) Reuters - Private investment...
  Classified to leaf with 23 documents
  Leaf question: Is this about business or finance?

================================================================================
SUMMARY
================================================================================
Total documents processed: 100
Unique leaf nodes reached: 8

Documents per leaf distribution:
  Leaf 140234567890123: 23 document(s)
  Leaf 140234567890456: 18 document(s)
  ...
```

### Requirements

- A pre-built RTP tree (can be created using `build_rtp_trees.py` or similar scripts)
- Dependencies from `pyproject.toml`:
  - `datasets>=2.0.0` (for HuggingFace datasets)
  - `transformers>=4.57.3` (for NLI models)
  - `torch>=2.9.1` (deep learning framework)
  - `numpy`, `tqdm`

### Notes

- The script uses the `query` function from `askme.rtp.query` module
- NLI model (`microsoft/deberta-v2-xlarge-mnli` by default) is used for answering questions at each tree node
- Inference time depends on document length, tree depth, and whether GPU acceleration is used
- For large datasets, consider using GPU acceleration with `--device cuda`
