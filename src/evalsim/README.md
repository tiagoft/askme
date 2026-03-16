# EvalSim

Evaluates the similarity within batches of texts.

### Example usage in code

```python
from evalsim.similarities import SimilarityCalculator

texts = ["first text", "second text", "third text"]

sc = SimilarityCalculator()
similarities = sc(texts)

print(f"Lexical: {similarities.lexical}")
print(f"Semantic: {similarities.semantic}")
print(f"Logical: {similarities.logical}")
```

### Example in command line

```bash
evalsim "This is one sentence" "This is another sentence" "This is a third sentence and so on so on"
```
