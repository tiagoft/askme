"""Tests for the query function."""

import pytest
from askme.rtp import RTPBuilder, TreeNode, query
from askme.askquestions import models


# Sample text collection for testing
sample_text_collection = [
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


def test_query_returns_leaf_node():
    """Test that query returns a leaf node."""
    # Build a tree
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    tree = builder(sample_text_collection)
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    # Query with a new document
    new_doc = "Puppies are young dogs and they love to play."
    leaf = query(new_doc, tree, nli_model, nli_tokenizer, device='cpu')
    
    # Verify it's a leaf node (no children)
    assert isinstance(leaf, TreeNode)
    assert leaf.left is None
    assert leaf.right is None
    assert len(leaf.documents) > 0


def test_query_with_document_similar_to_training():
    """Test query with a document similar to training documents."""
    # Build a tree
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    tree = builder(sample_text_collection)
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    # Query with a document similar to one in the collection
    new_doc = "Dogs are great pets that bark."
    leaf = query(new_doc, tree, nli_model, nli_tokenizer, device='cpu')
    
    # Should return a leaf node
    assert isinstance(leaf, TreeNode)
    assert leaf.left is None
    assert leaf.right is None


def test_query_with_different_document():
    """Test query with a document different from training documents."""
    # Build a tree
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    tree = builder(sample_text_collection)
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    # Query with a completely different document
    new_doc = "Python is a programming language used for data science."
    leaf = query(new_doc, tree, nli_model, nli_tokenizer, device='cpu')
    
    # Should still return a leaf node
    assert isinstance(leaf, TreeNode)
    assert leaf.left is None
    assert leaf.right is None


def test_query_with_single_node_tree():
    """Test query with a tree that has no children (single leaf)."""
    # Create a simple tree with no split
    simple_tree = TreeNode(
        documents=[0, 1, 2],
        question=None,
        left=None,
        right=None
    )
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    # Query should return the root itself (it's a leaf)
    new_doc = "Any document will work here."
    leaf = query(new_doc, simple_tree, nli_model, nli_tokenizer, device='cpu')
    
    # Should return the root (which is a leaf)
    assert leaf == simple_tree
    assert leaf.documents == [0, 1, 2]


def test_query_with_two_level_tree():
    """Test query with a manually constructed two-level tree."""
    # Create a two-level tree manually
    left_child = TreeNode(documents=[0, 1], question=None, left=None, right=None)
    right_child = TreeNode(documents=[2, 3], question=None, left=None, right=None)
    root = TreeNode(
        documents=[0, 1, 2, 3],
        question="Is this about cats?",
        left=left_child,
        right=right_child
    )
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    # Query with a cat-related document (should go left)
    cat_doc = "Cats are wonderful pets that meow."
    leaf = query(cat_doc, root, nli_model, nli_tokenizer, device='cpu')
    
    # Should be a leaf node
    assert isinstance(leaf, TreeNode)
    assert leaf.left is None
    assert leaf.right is None
    
    # Query with a non-cat document (should go right)
    other_doc = "Programming is fun."
    leaf2 = query(other_doc, root, nli_model, nli_tokenizer, device='cpu')
    
    # Should be a leaf node
    assert isinstance(leaf2, TreeNode)
    assert leaf2.left is None
    assert leaf2.right is None


def test_query_raises_error_on_none_tree():
    """Test that query raises ValueError when tree_root is None."""
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    with pytest.raises(ValueError, match="tree_root cannot be None"):
        query("Any document", None, nli_model, nli_tokenizer)


def test_query_raises_error_on_empty_tree():
    """Test that query raises ValueError when tree has no documents."""
    # Create a tree with no documents
    empty_tree = TreeNode(documents=[], question=None, left=None, right=None)
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    with pytest.raises(ValueError, match="must have at least one document"):
        query("Any document", empty_tree, nli_model, nli_tokenizer)


def test_query_with_custom_chunk_parameters():
    """Test query with custom chunk_size and overlap parameters."""
    # Build a tree
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=2)
    tree = builder(sample_text_collection)
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    # Query with custom chunk parameters
    new_doc = "Cats and dogs are both common household pets."
    leaf = query(
        new_doc,
        tree,
        nli_model,
        nli_tokenizer,
        device='cpu',
        chunk_size=100,
        overlap=10
    )
    
    # Should return a leaf node
    assert isinstance(leaf, TreeNode)
    assert leaf.left is None
    assert leaf.right is None


def test_query_consistency():
    """Test that querying the same document multiple times gives consistent results."""
    # Build a tree
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    tree = builder(sample_text_collection)
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    # Query the same document multiple times
    new_doc = "Dogs are loyal companions."
    leaf1 = query(new_doc, tree, nli_model, nli_tokenizer, device='cpu')
    leaf2 = query(new_doc, tree, nli_model, nli_tokenizer, device='cpu')
    leaf3 = query(new_doc, tree, nli_model, nli_tokenizer, device='cpu')
    
    # Should return the same leaf (same document set)
    assert leaf1.documents == leaf2.documents
    assert leaf2.documents == leaf3.documents


def test_query_with_long_document():
    """Test query with a long document that requires chunking."""
    # Build a tree
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    tree = builder(sample_text_collection)
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    # Create a long document
    long_doc = " ".join([
        "Dogs are amazing animals that have been companions to humans for thousands of years.",
        "They are known for their loyalty, intelligence, and ability to form strong bonds with their owners.",
        "Dogs come in many different breeds, each with unique characteristics and temperaments.",
        "Some dogs are great for families with children, while others are better suited for active individuals.",
        "Training a dog requires patience, consistency, and positive reinforcement.",
        "Dogs need regular exercise, proper nutrition, and veterinary care to stay healthy.",
    ])
    
    leaf = query(long_doc, tree, nli_model, nli_tokenizer, device='cpu', chunk_size=50, overlap=10)
    
    # Should return a leaf node
    assert isinstance(leaf, TreeNode)
    assert leaf.left is None
    assert leaf.right is None


def test_query_uses_existing_nli_functions():
    """Test that query uses the existing pool_nli_scores function."""
    # This is more of a code inspection test - we verify that our implementation
    # uses check_entailment.pool_nli_scores as required
    
    from askme.rtp.query import query as query_fn
    import inspect
    
    # Get the source code of the query function
    source = inspect.getsource(query_fn)
    
    # Verify it uses pool_nli_scores
    assert "pool_nli_scores" in source
    assert "check_entailment.pool_nli_scores" in source
    
    # Verify it uses check_entailment_nli
    assert "check_entailment.check_entailment_nli" in source


def test_query_multiple_documents_on_same_tree():
    """Test querying multiple different documents on the same tree."""
    # Build a tree
    builder = RTPBuilder(use_gpu=False, n_medoids=2, n_documents_to_answer=3)
    tree = builder(sample_text_collection)
    
    # Initialize NLI model
    nli_model, nli_tokenizer = models.make_nli_model()
    
    # Query with multiple different documents
    documents = [
        "Cats are independent creatures.",
        "Dogs love to play fetch.",
        "Eagles soar high in the mountains.",
        "Whales are massive marine mammals.",
    ]
    
    leaves = []
    for doc in documents:
        leaf = query(doc, tree, nli_model, nli_tokenizer, device='cpu')
        leaves.append(leaf)
        
        # Each should be a leaf node
        assert isinstance(leaf, TreeNode)
        assert leaf.left is None
        assert leaf.right is None
    
    # Should have gotten some results (not necessarily all different)
    assert len(leaves) == len(documents)
