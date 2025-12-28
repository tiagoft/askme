# Recursive thematic partitioning (RTP) module

# This module implements the Recursive Thematic Partitioning (RTP) algorithm
# for document segmentation based on thematic coherence.

import logging
from typing import Optional, List, Union

from .tree_models import TreeNode, TokenUsage
from .metrics import calculate_entropy, calculate_information_gain

from askme.makequestions.makequestion import make_a_question_about_collection
from askme.askquestions.check_entailment import check_entailment_nli

# Configure logging
logger = logging.getLogger(__name__)



def _rtp_recursion(
    documents: list[str],
    labels: Optional[List[Union[int, str]]],
    model_to_make_questions,
    model_to_answer_questions,
    tokenizer_to_answer_questions,
    current_depth: int,
    max_depth: int,
    min_documents_per_node: int = 2,
    parent: Optional[TreeNode] = None,
    verbose: bool = False,
) -> tuple[TreeNode, TokenUsage]:
    """
    Recursive helper function for RTP.
    """
    # Base case: if max depth reached or not enough documents, return leaf node
    if current_depth >= max_depth or len(documents) <= min_documents_per_node:
        no_usage = TokenUsage()
        leaf_node = TreeNode(documents=list(range(len(documents))), parent=parent)
        # Calculate entropy for leaf node if labels are provided
        if labels is not None and len(labels) > 0:
            leaf_node.entropy = calculate_entropy(labels)
        return leaf_node, no_usage
    
    # Calculate parent entropy before split
    parent_entropy = None
    if labels is not None and len(labels) > 0:
        parent_entropy = calculate_entropy(labels)
    
    # Generate a question about the collection
    result = make_a_question_about_collection(
        collection=documents,
        model=model_to_make_questions,
        retries=3,
    )
    
    question = result.output.hypothesis
    if verbose:
        print(f"Depth {current_depth}: Generated question: {question}")
        print(f"Documents: {documents}")
        if parent_entropy is not None:
            print(f"Parent Entropy: {parent_entropy:.4f}")
    
    usage = TokenUsage(
        total_tokens=result.usage().total_tokens,
        prompt_tokens=result.usage().input_tokens,
        completion_tokens=result.usage().output_tokens,
    )
    # Split documents based on the generated question
    left_docs = []
    right_docs = []
    left_labels = []
    right_labels = []
    
    for idx, doc in enumerate(documents):
        entailment_result = check_entailment_nli(
            model_to_answer_questions,
            tokenizer_to_answer_questions,
            premise=doc,
            hypothesis=question,
            device='cpu'
        )
        is_entailment = entailment_result[0]
        if is_entailment:
            left_docs.append(doc)
            if labels is not None:
                left_labels.append(labels[idx])
        else:
            right_docs.append(doc)
            if labels is not None:
                right_labels.append(labels[idx])
    
    # Calculate information gain if labels are provided
    information_gain = None
    if labels is not None and len(labels) > 0:
        information_gain = calculate_information_gain(labels, left_labels, right_labels)
        
        # Log information gain to terminal
        log_message = (f"Depth {current_depth}: Information Gain = {information_gain:.4f} "
                      f"(Entropy before: {parent_entropy:.4f}, "
                      f"Left: {len(left_labels)}/{len(labels)}, "
                      f"Right: {len(right_labels)}/{len(labels)})")
        print(log_message)
        logger.info(log_message)
    
    # Create current node
    current_node = TreeNode(
        documents=list(range(len(documents))),
        question=question,
        parent=parent,
        entropy=parent_entropy,
        information_gain=information_gain
    )
    
    # Recurse on left and right splits
    left_node, usage_left = _rtp_recursion(
        left_docs,
        left_labels if labels is not None else None,
        model_to_make_questions,
        model_to_answer_questions,
        tokenizer_to_answer_questions,
        current_depth + 1,
        max_depth,
        parent=current_node,
        verbose=verbose,
    )
    current_node.left = left_node
    
    right_node, usage_right = _rtp_recursion(
        right_docs,
        right_labels if labels is not None else None,
        model_to_make_questions,
        model_to_answer_questions,
        tokenizer_to_answer_questions,
        current_depth + 1,
        max_depth,
        parent=current_node,
        verbose=verbose,
    )
    current_node.right = right_node
    
    usage.total_tokens += usage_left.total_tokens + usage_right.total_tokens
    usage.prompt_tokens += usage_left.prompt_tokens + usage_right.prompt_tokens
    usage.completion_tokens += usage_left.completion_tokens + usage_right.completion_tokens
    
    return current_node, usage


def rtp(
    documents: list[str],
    model_to_make_questions,
    model_to_answer_questions,
    tokenizer_to_answer_questions,
    max_depth: int = 3,
    labels: Optional[List[Union[int, str]]] = None,
    verbose: bool = False,
    log_file: Optional[str] = None,
) -> tuple[TreeNode, TokenUsage]:
    """
    Perform Recursive Thematic Partitioning on a list of documents.
    
    Args:
        documents (list[str]): List of document texts.
        model_to_make_questions: Model for generating questions.
        model_to_answer_questions: Model for answering questions (NLI).
        tokenizer_to_answer_questions: Tokenizer for the answer model.
        max_depth (int): Maximum depth of the partitioning tree.
        labels (Optional[List]): Ground-truth labels for entropy/IG calculation.
        verbose (bool): Whether to print detailed information.
        log_file (Optional[str]): Path to log file for IG values.
        
    Returns:
        tuple[TreeNode, TokenUsage]: Root of the RTP tree and token usage.
    """
    # Configure file logging if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO)
    
    root, usage = _rtp_recursion(
        documents,
        labels,
        model_to_make_questions,
        model_to_answer_questions,
        tokenizer_to_answer_questions,
        current_depth=0,
        max_depth=max_depth,
        parent=None,
        verbose=verbose,
    )
    
    # Clean up file handler
    if log_file:
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                handler.close()
                logger.removeHandler(handler)
    
    return root, usage

def print_tree(node: TreeNode, depth: int = 0):
    """
    Utility function to print the RTP tree structure.
    """
    indent = "  " * depth
    print(f"{indent}Node Depth {depth}: Documents {node.documents}")
    if node.question:
        print(f"{indent}  Question: {node.question}")
    if node.entropy is not None:
        print(f"{indent}  Entropy: {node.entropy:.4f}")
    if node.information_gain is not None:
        print(f"{indent}  Information Gain: {node.information_gain:.4f}")
    if node.left:
        print_tree(node.left, depth + 1)
    if node.right:
        print_tree(node.right, depth + 1)