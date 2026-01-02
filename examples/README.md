# RTPBuilder Examples

This directory contains examples showing how to use the RTPBuilder class.

## Basic Usage

See `use_rtp_builder.py` for a complete example of:
- Initializing the RTPBuilder
- Building an RTP tree from a text collection
- Inspecting the resulting tree structure
- Saving the tree to JSON
- Reusing the builder for multiple collections

## Running the Example

```bash
python examples/use_rtp_builder.py
```

Note: Make sure you have:
1. All dependencies installed
2. API credentials set up (for the LLM model)
3. Either a GPU with CUDA support, or set `use_gpu=False` in the builder

## Key Points

- **Initialization overhead**: The RTPBuilder initializes all models (embedding, NLI, LLM) once in `__init__`. This is expensive but only needs to be done once.

- **Fast execution**: The `__call__` method can be called multiple times on different collections without reinitializing models.

- **Flexibility**: The builder accepts any iterable of strings (lists, generators, PyTorch datasets, etc.)

- **GPU support**: Set `use_gpu=True` for GPU acceleration (requires CUDA)
