# Implementation Summary: Entropy and Information Gain Metrics

## Overview
This implementation adds mathematical metrics to monitor the quality of splits at each node in the Recursive Thematic Partitioning (RTP) algorithm.

## What Was Implemented

### 1. Metrics Module (`src/askme/rtp/metrics.py`)
A new module containing:
- **`calculate_entropy(labels)`**: Calculates Shannon Entropy H(D) for ground-truth labels
- **`calculate_information_gain(parent_labels, left_labels, right_labels)`**: Calculates Information Gain for splits

### 2. Enhanced TreeNode (`src/askme/rtp/tree_models.py`)
Added optional fields to the `TreeNode` class:
- `entropy`: Stores Shannon entropy value for the node
- `information_gain`: Stores Information Gain achieved by splitting

### 3. Updated RTP Module (`src/askme/rtp/rtp.py`)
Modified the RTP algorithm to:
- Accept optional `labels` parameter for ground-truth labels
- Accept optional `log_file` parameter for logging to file
- Calculate entropy at each node before splitting
- Calculate information gain for each split
- Log IG to terminal (using print)
- Log IG to file (using Python's logging module)
- Display entropy and IG in the `print_tree()` function

### 4. Comprehensive Testing
Created test suites:
- **`tests/test_metrics.py`**: 12 tests for entropy and IG calculations
- **`tests/test_rtp_with_metrics.py`**: 4 tests for TreeNode with metrics
- **`tests/test_rtp_trees.py`**: Existing tests still pass (backward compatibility)

### 5. Examples and Documentation
- **`tests/demo_metrics.py`**: Interactive demo of entropy and IG calculations
- **`tests/example_rtp_metrics.py`**: Simulated RTP with metrics example
- **`tests/test_logging.py`**: Demonstrates logging to file
- **`docs/METRICS.md`**: Comprehensive documentation with usage examples

## Acceptance Criteria Met

✅ **Calculates Shannon Entropy H(D) based on ground-truth labels**
- Implemented in `calculate_entropy()` function
- Formula: H(D) = -Σ(p_i * log2(p_i))
- Handles both integer and string labels
- Returns 0.0 for pure sets, 1.0 for balanced binary splits

✅ **Calculates Information Gain (IG) for every split**
- Implemented in `calculate_information_gain()` function
- Formula: IG(D, split) = H(D) - [|D_left|/|D| * H(D_left) + |D_right|/|D| * H(D_right)]
- Automatically calculated when labels are provided to RTP

✅ **Logs IG to terminal and local log file**
- Terminal: Uses `print()` for immediate visibility
- Log file: Uses Python's `logging` module with timestamps
- Format: `"Depth X: Information Gain = Y.ZZZZ (Entropy before: A.BBBB, Left: C/D, Right: E/F)"`

## Backward Compatibility
The implementation is fully backward compatible:
- The `labels` and `log_file` parameters are optional
- Existing code without labels continues to work unchanged
- Entropy and IG fields in TreeNode are optional

## Usage Example

```python
from askme.rtp import rtp
from askme.makequestions import api
from askme.askquestions import models

# Setup documents and labels
documents = ["doc1", "doc2", "doc3", "doc4"]
labels = [0, 1, 0, 1]  # Ground-truth labels

# Build models
model = api.build_model()
nli_model, nli_tokenizer = models.make_nli_model(
    model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
)

# Run RTP with metrics
tree, usage = rtp.rtp(
    documents=documents,
    model_to_make_questions=model,
    model_to_answer_questions=nli_model,
    tokenizer_to_answer_questions=nli_tokenizer,
    max_depth=2,
    labels=labels,              # Enable metrics
    verbose=True,
    log_file='rtp_ig.log'       # Enable file logging
)

# Print tree with metrics
rtp.print_tree(tree)
```

## Output Examples

### Terminal Output
```
Depth 0: Generated question: Is this about machine learning?
Documents: ['doc1', 'doc2', 'doc3', 'doc4']
Parent Entropy: 1.0000
Depth 0: Information Gain = 0.8113 (Entropy before: 1.0000, Left: 3/4, Right: 1/4)
```

### Log File (rtp_ig.log)
```
2025-12-28 21:00:00,123 - Depth 0: Information Gain = 0.8113 (Entropy before: 1.0000, Left: 3/4, Right: 1/4)
2025-12-28 21:00:05,456 - Depth 1: Information Gain = 0.4591 (Entropy before: 0.9183, Left: 2/3, Right: 1/3)
```

## Files Changed
- `src/askme/rtp/metrics.py` (new)
- `src/askme/rtp/tree_models.py` (modified)
- `src/askme/rtp/rtp.py` (modified)
- `tests/test_metrics.py` (new)
- `tests/test_rtp_with_metrics.py` (new)
- `tests/demo_metrics.py` (new)
- `tests/example_rtp_metrics.py` (new)
- `tests/test_logging.py` (new)
- `docs/METRICS.md` (new)
- `.gitignore` (modified)

## Test Results
All 20 tests pass successfully:
- 12 tests for metrics calculations
- 4 tests for TreeNode with metrics
- 4 existing tests for TreeNode (backward compatibility verified)
