# EvalSim

Evaluates the similarity within batches of texts.

Example usage:

```python
from evalsim.similarities import SimilarityCalculator

texts = ["first text", "second text", "third text"]

sc = SimilarityCalculator()
similarities = sc(texts)

print(f"Lexical: {similarities.lexical}")
print(f"Semantic: {similarities.semantic}")
print(f"Logical: {similarities.logical}")
```