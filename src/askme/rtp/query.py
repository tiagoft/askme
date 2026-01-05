"""
Query function for navigating RTP trees using NLI.

This module provides functionality to classify a new document into an RTP tree
by using Natural Language Inference (NLI) to answer questions at each node.
"""

from typing import Optional
from .tree_models import TreeNode
from ..askquestions import check_entailment


def query(
    document: str,
    tree_root: TreeNode,
    nli_model,
    nli_tokenizer,
    device: str = 'cpu',
    chunk_size: int = 200,
    overlap: int = 20,
) -> TreeNode:
    """
    Query which leaf node a document belongs to in an RTP tree.
    
    This function navigates through the tree by using NLI to answer the question
    at each internal node. If the document entails the question (answers YES),
    it follows the left child; otherwise, it follows the right child. The process
    continues until a leaf node is reached.
    
    Args:
        document: The text document to classify
        tree_root: The root TreeNode of the RTP tree
        nli_model: The NLI model for checking entailment
        nli_tokenizer: Tokenizer for the NLI model
        device: Device to run NLI on ('cpu' or 'cuda')
        chunk_size: Size of text chunks for NLI processing (in words)
        overlap: Overlap between chunks (in words)
        
    Returns:
        The leaf TreeNode where the document should be classified
        
    Raises:
        ValueError: If tree_root is None or has no documents
        
    Example:
        >>> from askme.rtp import RTPBuilder, query
        >>> from askme.askquestions import models
        >>> 
        >>> # Build a tree
        >>> builder = RTPBuilder(use_gpu=False)
        >>> texts = ["Dogs bark.", "Cats meow.", "Birds fly."]
        >>> tree = builder(texts)
        >>> 
        >>> # Query a new document
        >>> nli_model, nli_tokenizer = models.make_nli_model()
        >>> new_doc = "Puppies are young dogs."
        >>> leaf = query(new_doc, tree, nli_model, nli_tokenizer)
        >>> print(f"Document classified to leaf with {len(leaf.documents)} docs")
    """
    if tree_root is None:
        raise ValueError("tree_root cannot be None")
    
    if not tree_root.documents:
        raise ValueError("tree_root must have at least one document")
    
    current_node = tree_root
    
    # Traverse the tree until we reach a leaf node
    while current_node.left is not None or current_node.right is not None:
        # Internal node - must have a question
        if current_node.question is None:
            # If there's no question but there are children, this is an error
            raise ValueError(f"Internal node has children but no question")
        
        # Use NLI to answer the question
        hypothesis = current_node.question
        pooled_results = check_entailment.pool_nli_scores(
            check_fn=check_entailment.check_entailment_nli,
            premise=document,
            hypothesis=hypothesis,
            chunk_size=chunk_size,
            overlap=overlap,
            model=nli_model,
            tokenizer=nli_tokenizer,
            device=device,
        )
        entails, entailment_score, contradiction_score, P_entailment = pooled_results
        
        # Navigate based on entailment
        if entails:
            # Document entails the hypothesis (YES) -> go left
            if current_node.left is not None:
                current_node = current_node.left
            else:
                # Left child doesn't exist, stay at current node (it's a leaf)
                break
        else:
            # Document does not entail the hypothesis (NO) -> go right
            if current_node.right is not None:
                current_node = current_node.right
            else:
                # Right child doesn't exist, stay at current node (it's a leaf)
                break
    
    # Return the leaf node
    return current_node
