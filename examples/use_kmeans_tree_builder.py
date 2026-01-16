"""
Example usage of KMeansTreeBuilder class.

This script demonstrates how to use the KMeansTreeBuilder to create a tree
using K-means clustering (k=2) from a collection of text documents.
"""

from askme.rtp import KMeansTreeBuilder

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
]


def main():
    """
    Example of using KMeansTreeBuilder.
    
    The KMeansTreeBuilder uses K-means clustering with k=2 to split documents,
    then generates a hypothesis that distinguishes between the two clusters.
    """
    
    # Step 1: Initialize the KMeansTreeBuilder with your preferred settings
    print("Initializing KMeansTreeBuilder...")
    builder = KMeansTreeBuilder(
        use_gpu=False,  # Set to True if you have GPU available
        n_medoids_per_cluster=2,  # Select 2 medoids from each cluster
        n_documents_to_answer=6,
    )
    print("KMeansTreeBuilder initialized successfully!")
    
    # Step 2: Call the builder with your text collection
    print("\nBuilding K-means tree...")
    tree_root = builder(text_collection)
    
    # Step 3: Inspect the resulting tree
    print("\n=== K-means Tree Results ===")
    print(f"Question: {tree_root.question}")
    print(f"Total documents: {len(tree_root.documents)}")
    
    if tree_root.left is not None:
        print(f"\nLeft child (answers YES to question):")
        print(f"  Documents: {tree_root.left.documents}")
        print(f"  Texts: {[text_collection[i] for i in tree_root.left.documents[:3]]}...")
    
    if tree_root.right is not None:
        print(f"\nRight child (answers NO to question):")
        print(f"  Documents: {tree_root.right.documents}")
        print(f"  Texts: {[text_collection[i] for i in tree_root.right.documents[:3]]}...")
    
    # Step 4 (Optional): Save the tree to JSON
    print("\n=== Saving tree to JSON ===")
    json_string = tree_root.model_dump_json()
    with open('kmeans_tree.json', 'w') as f:
        f.write(json_string)
    print("Tree saved to kmeans_tree.json")
    
    # Step 5 (Optional): Call the builder again with a different collection
    print("\n=== Using builder on a different collection ===")
    new_collection = [
        "Python is a programming language.",
        "Java is used for enterprise software.",
        "JavaScript runs in browsers.",
        "The ocean is vast and deep.",
        "Mountains reach high into the sky.",
    ]
    
    tree_root_2 = builder(new_collection)
    print(f"Second tree question: {tree_root_2.question}")
    print("Successfully created second tree using the same builder!")


if __name__ == "__main__":
    main()
