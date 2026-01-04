"""
Example usage of RTPBuilder with retry mechanism.

This script demonstrates how to use the RTPBuilder with retry parameters
to ensure generated questions lead to acceptable split ratios.
"""

from askme.rtp import RTPBuilder

# Sample text collection with clear split potential
text_collection = [
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
    """
    Example of using RTPBuilder with retry mechanism.
    
    The retry mechanism helps ensure that generated questions lead to
    splits that are within acceptable ratio bounds.
    """
    
    print("=" * 60)
    print("RTPBuilder with Retry Mechanism Example")
    print("=" * 60)
    
    # Example 1: Basic usage with retry parameters
    print("\n1. Basic usage with retry parameters")
    print("-" * 60)
    
    builder = RTPBuilder(
        use_gpu=False,
        n_medoids=4,
        n_documents_to_answer=6,
        max_retries=3,  # Allow up to 3 retries
        min_split_ratio=0.2,  # Minimum 20% of documents in smaller child
        max_split_ratio=0.8,  # Maximum 80% (i.e., at least 20% in each)
    )
    
    print(f"Configuration:")
    print(f"  - max_retries: {builder.max_retries}")
    print(f"  - min_split_ratio: {builder.min_split_ratio}")
    print(f"  - max_split_ratio: {builder.max_split_ratio}")
    
    tree_root, metrics = builder(text_collection, return_metrics=True)
    
    print(f"\nResults:")
    print(f"  - Question: {tree_root.question}")
    print(f"  - Split ratio: {metrics.split_ratio:.2f}")
    print(f"  - LLM input tokens: {metrics.llm_input_tokens}")
    print(f"  - LLM output tokens: {metrics.llm_output_tokens}")
    print(f"  - NLI calls: {metrics.nli_calls}")
    
    if tree_root.left and tree_root.right:
        print(f"  - Left child documents: {len(tree_root.left.documents)}")
        print(f"  - Right child documents: {len(tree_root.right.documents)}")
    
    # Example 2: Very restrictive split ratio (may trigger retries)
    print("\n2. Very restrictive split ratio constraints")
    print("-" * 60)
    
    builder_strict = RTPBuilder(
        use_gpu=False,
        n_medoids=3,
        n_documents_to_answer=4,
        max_retries=5,  # Allow more retries
        min_split_ratio=0.4,  # Very restrictive: 40-60% split
        max_split_ratio=0.6,
    )
    
    print(f"Configuration:")
    print(f"  - max_retries: {builder_strict.max_retries}")
    print(f"  - min_split_ratio: {builder_strict.min_split_ratio}")
    print(f"  - max_split_ratio: {builder_strict.max_split_ratio}")
    
    tree_root_2, metrics_2 = builder_strict(text_collection, return_metrics=True)
    
    print(f"\nResults:")
    print(f"  - Question: {tree_root_2.question}")
    print(f"  - Split ratio: {metrics_2.split_ratio:.2f}")
    print(f"  - LLM input tokens: {metrics_2.llm_input_tokens}")
    print(f"  - LLM output tokens: {metrics_2.llm_output_tokens}")
    
    if tree_root_2.left and tree_root_2.right:
        print(f"  - Left child documents: {len(tree_root_2.left.documents)}")
        print(f"  - Right child documents: {len(tree_root_2.right.documents)}")
    
    # Example 3: No split ratio constraints (original behavior)
    print("\n3. No split ratio constraints (default behavior)")
    print("-" * 60)
    
    builder_no_constraints = RTPBuilder(
        use_gpu=False,
        n_medoids=3,
        n_documents_to_answer=4,
        # No min_split_ratio or max_split_ratio specified
    )
    
    print(f"Configuration:")
    print(f"  - max_retries: {builder_no_constraints.max_retries}")
    print(f"  - min_split_ratio: {builder_no_constraints.min_split_ratio}")
    print(f"  - max_split_ratio: {builder_no_constraints.max_split_ratio}")
    
    tree_root_3, metrics_3 = builder_no_constraints(text_collection, return_metrics=True)
    
    print(f"\nResults:")
    print(f"  - Question: {tree_root_3.question}")
    print(f"  - Split ratio: {metrics_3.split_ratio:.2f}")
    print(f"  - LLM input tokens: {metrics_3.llm_input_tokens}")
    
    print("\n" + "=" * 60)
    print("Key Observations:")
    print("=" * 60)
    print("""
1. When split ratio constraints are specified, the builder will retry
   generating questions if the split ratio is outside the acceptable range.
   
2. Each retry uses a blacklist to avoid regenerating the same question.

3. The blacklist is passed to the LLM to request different questions.

4. If all retries are exhausted, the builder returns the last result
   even if the split ratio is still outside the bounds.

5. When no constraints are specified (None), the builder accepts any split
   and does not perform retries (original behavior preserved).
    """)


if __name__ == "__main__":
    main()
