
print("Importing")
from evalsim.similarities import SimilarityCalculator

print("Initializing SimilarityCalculator")
sc = SimilarityCalculator(
    use_logical=False,
    use_semantic=True,
) 

print("Calculating similarities")
texts = ["I like the taste of water", "I like water", "I like to drink water"]
similarities = sc(texts)

print(f"Lexical: {similarities.lexical}")
print(f"Semantic: {similarities.semantic}")
print(f"Logical: {similarities.logical}")

texts = ["The sky is blue", "The sky is red", "My car is red"]
similarities = sc(texts)
print(f"Lexical: {similarities.lexical}")
print(f"Semantic: {similarities.semantic}")
print(f"Logical: {similarities.logical}")
