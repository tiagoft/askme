"""
Example usage of the query function.

This script demonstrates how to use the query function to classify a new document
into an existing RTP tree.
"""

from askme.rtp import RTPBuilder, query
from askme.askquestions import models


def main():
    """
    Example of using the query function to classify new documents.
    """
    
    # Sample text collection to build a tree
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
    
    print("=" * 60)
    print("Query Function Example")
    print("=" * 60)
    
    # Step 1: Build an RTP tree
    print("\n1. Building RTP tree from training documents...")
    builder = RTPBuilder(
        use_gpu=False,
        n_medoids=4,
        n_documents_to_answer=6,
    )
    tree = builder(text_collection)
    print(f"   Tree built with question: '{tree.question}'")
    
    if tree.left:
        print(f"   - Left child has {len(tree.left.documents)} documents")
    if tree.right:
        print(f"   - Right child has {len(tree.right.documents)} documents")
    
    # Step 2: Initialize NLI model for querying
    print("\n2. Loading NLI model...")
    nli_model, nli_tokenizer = models.make_nli_model()
    print("   NLI model loaded successfully!")
    
    # Step 3: Query new documents
    print("\n3. Classifying new documents using the query function...")
    
    new_documents = [
        "Puppies are young dogs that love to play.",
        "Kittens are baby cats.",
        "Eagles soar through the clouds.",
        "Sharks are dangerous predators in the ocean.",
    ]
    
    for i, new_doc in enumerate(new_documents):
        print(f"\n   Document {i+1}: '{new_doc}'")
        
        # Use query function to find the leaf node
        leaf = query(
            document=new_doc,
            tree_root=tree,
            nli_model=nli_model,
            nli_tokenizer=nli_tokenizer,
            device='cpu'
        )
        
        print(f"   -> Classified to leaf with {len(leaf.documents)} documents:")
        
        # Show which training documents are in the same leaf
        similar_docs = [text_collection[idx] for idx in leaf.documents[:3]]
        for doc in similar_docs:
            print(f"      * {doc}")
        if len(leaf.documents) > 3:
            print(f"      ... and {len(leaf.documents) - 3} more")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
