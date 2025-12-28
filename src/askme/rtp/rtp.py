# Recursive thematic partitioning (RTP) module

# This module implements the Recursive Thematic Partitioning (RTP) algorithm
# for document segmentation based on thematic coherence.

from .tree_models import TreeNode, TokenUsage

from askme.makequestions.makequestion import make_a_question_about_collection
from askme.askquestions.check_entailment import check_entailment_nli



def _rtp_recursion(
    documents: list[str],
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
        return TreeNode(documents=list(range(len(documents))), parent=parent), no_usage
    
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
    
    usage = TokenUsage(
        total_tokens=result.usage().total_tokens,
        prompt_tokens=result.usage().input_tokens,
        completion_tokens=result.usage().output_tokens,
    )
    # Split documents based on the generated question
    left_docs = []
    right_docs = []
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
        else:
            right_docs.append(doc)
    
    # Create current node
    current_node = TreeNode(
        documents=list(range(len(documents))),
        question=question,
        parent=parent
    )
    
    # Recurse on left and right splits
    left_node, usage_left = _rtp_recursion(
        left_docs,
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
    verbose: bool = False,
) -> tuple[TreeNode, TokenUsage]:
    """
    Perform Recursive Thematic Partitioning on a list of documents no optimization.
    
    Args:
        documents (list[str]): List of document texts.
        max_depth (int): Maximum depth of the partitioning tree.
        
    Returns:
        TreeNode: Root of the RTP tree.
    """
    root, usage = _rtp_recursion(
        documents,
        model_to_make_questions,
        model_to_answer_questions,
        tokenizer_to_answer_questions,
        current_depth=0,
        max_depth=max_depth,
        parent=None,
        verbose=verbose,
    )
    return root, usage

def print_tree(node: TreeNode, depth: int = 0):
    """
    Utility function to print the RTP tree structure.
    """
    indent = "  " * depth
    print(f"{indent}Node Depth {depth}: Documents {node.documents}")
    if node.question:
        print(f"{indent}  Question: {node.question}")
    if node.left:
        print_tree(node.left, depth + 1)
    if node.right:
        print_tree(node.right, depth + 1)