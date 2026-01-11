"""
Example usage of RTPRecursion class.

This script demonstrates how to use RTPRecursion to create a recursive RTP tree
from a collection of text documents with stopping criteria.
"""

from askme.rtp import RTPBuilder, RTPRecursion

# Sample text collection
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
    "Tigers are powerful predators.",
    "Pandas eat bamboo.",
    "Whales live in the ocean.",
    "Dolphins are intelligent mammals.",
    "Penguins cannot fly but swim well.",
]


def print_tree(node, text_collection, depth=0, prefix=""):
    """Recursively print the tree structure."""
    indent = "  " * depth
    
    # Print current node info
    print(f"{indent}{prefix}Node with {len(node.documents)} documents")
    if node.question:
        print(f"{indent}  Question: {node.question}")
    
    # Print sample documents at this node
    sample_docs = [text_collection[i] for i in node.documents[:3]]
    for i, doc in enumerate(sample_docs):
        print(f"{indent}  - {doc}")
    if len(node.documents) > 3:
        print(f"{indent}  ... and {len(node.documents) - 3} more documents")
    
    # Print metrics if available
    if node.metrics:
        print(f"{indent}  Split ratio: {node.metrics.split_ratio:.2f}")
        print(f"{indent}  NLI calls: {node.metrics.nli_calls}")
    
    # Recurse into children
    if node.left:
        print(f"{indent}  LEFT (YES):")
        print_tree(node.left, text_collection, depth + 1, "L:")
    
    if node.right:
        print(f"{indent}  RIGHT (NO):")
        print_tree(node.right, text_collection, depth + 1, "R:")


def main():
    """
    Example of using RTPRecursion.
    
    RTPRecursion automatically builds a complete tree by recursively applying
    RTPBuilder with stopping criteria.
    """
    
    # Step 1: Initialize the RTPBuilder
    print("Initializing RTPBuilder...")
    builder = RTPBuilder(
        use_gpu=False,
        n_medoids=2,
        n_documents_to_answer=6,
    )
    print("RTPBuilder initialized successfully!")
    
    # Step 2: Initialize RTPRecursion with stopping criteria
    print("\nInitializing RTPRecursion...")
    recursion = RTPRecursion(
        builder=builder,
        min_node_size=3,      # Don't split nodes with fewer than 3 documents
        min_split_ratio=0.05,  # Split should have at least 20% in smaller child
        max_split_ratio=0.95,  # Split should have at most 80% in smaller child
        max_depth=3,          # Maximum tree depth
    )
    print("RTPRecursion initialized successfully!")
    
    # Step 3: Build the recursive tree
    print("\nBuilding recursive RTP tree...")
    tree_root, global_metrics = recursion(text_collection)
    
    # Step 4: Display the tree structure
    print("\n=== Recursive RTP Tree Structure ===")
    print_tree(tree_root, text_collection)
    
    # Step 5: Display global metrics
    print("\n=== Global Metrics (Accumulated Across All Nodes) ===")
    print(f"Total LLM Input Tokens: {global_metrics.llm_input_tokens}")
    print(f"Total LLM Output Tokens: {global_metrics.llm_output_tokens}")
    print(f"Total NLI Calls: {global_metrics.nli_calls}")
    print(f"Total FAISS Search Time: {global_metrics.faiss_search_time_ms:.2f} ms")
    print(f"Total Label Propagation Time: {global_metrics.label_propagation_time_ms:.2f} ms")
    print(f"Total Time: {global_metrics.total_time_ms:.2f} ms")
    print(f"Total LLM Request Time: {global_metrics.llm_request_time_ms:.2f} ms")
    print(f"Total NLI Time: {global_metrics.nli_time_ms:.2f} ms")
    print(f"Number of Nodes: {global_metrics.num_nodes}")
    
    # Compute averages for metrics that should be averaged
    if global_metrics.num_nodes > 0:
        avg_split_ratio = global_metrics.split_ratio / global_metrics.num_nodes
        avg_nli_confidence = global_metrics.medoid_nli_confidence_avg / global_metrics.num_nodes
        print(f"Average Split Ratio: {avg_split_ratio:.2f}")
        print(f"Average Medoid NLI Confidence: {avg_nli_confidence:.2f}")
    
    # Step 6: Save the tree to JSON
    print("\n=== Saving tree to JSON ===")
    json_string = tree_root.model_dump_json()
    with open('rtp_recursive_tree.json', 'w') as f:
        f.write(json_string)
    print("Tree saved to rtp_recursive_tree.json")
    
    # Step 7: Demonstrate tree traversal to find a document
    print("\n=== Using the Tree for Search ===")
    query_doc_index = 5  # "The dog is in the yard."
    print(f"Searching for document {query_doc_index}: {text_collection[query_doc_index]}")
    
    def find_leaf(node, doc_index):
        """Find the leaf node containing a document."""
        if node.left is None and node.right is None:
            # This is a leaf
            if doc_index in node.documents:
                return node
            return None
        
        # Check children
        if node.left and doc_index in node.left.documents:
            return find_leaf(node.left, doc_index)
        elif node.right and doc_index in node.right.documents:
            return find_leaf(node.right, doc_index)
        
        return None
    
    leaf = find_leaf(tree_root, query_doc_index)
    if leaf:
        print(f"Document found in leaf with {len(leaf.documents)} documents:")
        for i in leaf.documents:
            print(f"  - {text_collection[i]}")


if __name__ == "__main__":
    main()
