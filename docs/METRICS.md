# Entropy and Information Gain Metrics

This module provides mathematical metrics to monitor the quality of splits in the Recursive Thematic Partitioning (RTP) algorithm.

## Features

### Shannon Entropy
Calculates the Shannon Entropy H(D) for a set of labels based on ground-truth data:
```
H(D) = -Σ(p_i * log2(p_i))
```
where p_i is the proportion of class i in the dataset.

- **Pure set** (all same class): Entropy = 0.0
- **Maximum uncertainty** (balanced binary): Entropy = 1.0
- **Multiple classes**: Higher entropy indicates more mixing

### Information Gain
Calculates the Information Gain (IG) for a split, measuring the reduction in entropy:
```
IG(D, split) = H(D) - [|D_left|/|D| * H(D_left) + |D_right|/|D| * H(D_right)]
```

- **Perfect split** (complete separation): IG equals parent entropy
- **No improvement**: IG = 0.0
- **Partial improvement**: 0 < IG < parent entropy

### Logging
- **Terminal output**: Prints IG values during tree construction
- **Log file**: Saves IG values with timestamps to a specified file for later analysis

## Usage

### Basic Metrics Calculation

```python
from askme.rtp.metrics import calculate_entropy, calculate_information_gain

# Calculate entropy
labels = [0, 0, 1, 1]
entropy = calculate_entropy(labels)
print(f"Entropy: {entropy:.4f}")  # Output: 1.0000

# Calculate information gain
parent_labels = [0, 0, 1, 1]
left_labels = [0, 0]
right_labels = [1, 1]
ig = calculate_information_gain(parent_labels, left_labels, right_labels)
print(f"Information Gain: {ig:.4f}")  # Output: 1.0000 (perfect split)
```

### RTP with Metrics

```python
from askme.rtp import rtp
from askme.makequestions import api
from askme.askquestions import models

# Prepare your documents and labels
documents = [
    "The cat sat on the mat.",
    "The dog barked loudly.",
    "I like cats.",
    "The dog is in the yard.",
]
labels = [0, 1, 0, 1]  # Ground-truth labels

# Build models
model = api.build_model()
nli_model, nli_tokenizer = models.make_nli_model(
    model_name='MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'
)

# Run RTP with metrics and logging
tree, usage = rtp.rtp(
    documents=documents,
    model_to_make_questions=model,
    model_to_answer_questions=nli_model,
    tokenizer_to_answer_questions=nli_tokenizer,
    max_depth=2,
    labels=labels,  # Provide labels for metrics
    verbose=True,
    log_file='rtp_information_gain.log'  # Save to log file
)

# Print tree with metrics
rtp.print_tree(tree)
```

### TreeNode with Metrics

The `TreeNode` class now includes optional fields for storing metrics:

```python
from askme.rtp.tree_models import TreeNode

node = TreeNode(
    documents=[0, 1, 2, 3],
    question="Is this about machine learning?",
    entropy=1.0,
    information_gain=0.5
)
```

## Output Examples

### Terminal Output
```
Depth 0: Generated question: Is this about cats?
Documents: ['The cat sat on the mat.', 'The dog barked loudly.', ...]
Parent Entropy: 1.0000
Depth 0: Information Gain = 0.8113 (Entropy before: 1.0000, Left: 3/4, Right: 1/4)
```

### Log File Format
```
2025-12-28 21:00:00,123 - Depth 0: Information Gain = 0.8113 (Entropy before: 1.0000, Left: 3/4, Right: 1/4)
2025-12-28 21:00:05,456 - Depth 1: Information Gain = 0.4591 (Entropy before: 0.9183, Left: 2/3, Right: 1/3)
```

## API Reference

### `calculate_entropy(labels: List[Union[int, str]]) -> float`
Calculate Shannon Entropy for a set of labels.

**Parameters:**
- `labels`: List of ground-truth labels (can be integers or strings)

**Returns:**
- `float`: Shannon entropy value (0.0 for empty or homogeneous sets)

### `calculate_information_gain(parent_labels, left_labels, right_labels) -> float`
Calculate Information Gain for a split.

**Parameters:**
- `parent_labels`: Labels in the parent node before split
- `left_labels`: Labels in the left child node after split
- `right_labels`: Labels in the right child node after split

**Returns:**
- `float`: Information gain value (non-negative)

### `rtp(..., labels=None, log_file=None)`
Perform Recursive Thematic Partitioning with optional metrics.

**New Parameters:**
- `labels` (Optional[List]): Ground-truth labels for entropy/IG calculation
- `log_file` (Optional[str]): Path to log file for IG values

## Testing

Run the test suite:
```bash
pytest tests/test_metrics.py tests/test_rtp_with_metrics.py -v
```

Run demo scripts:
```bash
python tests/demo_metrics.py
python tests/example_rtp_metrics.py
python tests/test_logging.py
```

## Implementation Details

- **Backward Compatible**: The metrics are optional. Existing code without labels will continue to work.
- **Efficient**: Entropy and IG calculations are performed in O(n) time.
- **Flexible**: Works with both integer and string labels.
- **Logging**: Uses Python's standard logging module with optional file output.
