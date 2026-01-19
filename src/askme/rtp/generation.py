from pydantic import BaseModel
from .tree_models import TreeNode, TreeDecision, TreePath
import numpy as np

   
def path_to_dict(path: TreePath) -> list[dict]:
    """Convert TreePath to a list of dictionaries for easier serialization."""
    return [ {decision.hypothesis: decision.decision} for decision in path.decisions ]

def dict_to_path(path_dicts: list[dict]) -> TreePath:
    """Convert a list of dictionaries back to a TreePath."""
    decisions = []
    for decision_dict in path_dicts:
        for hypothesis, decision in decision_dict.items():
            decisions.append(TreeDecision(hypothesis=hypothesis, decision=decision))
    return TreePath(decisions=decisions)

def paths_are_equal(path1: TreePath, path2: TreePath) -> tuple[bool, int]:
    """Check if two TreePaths are equal."""
    equal_levels = 0
    if len(path1.decisions) != len(path2.decisions):
        return False, equal_levels
    for dec1, dec2 in zip(path1.decisions, path2.decisions):
        if dec1.hypothesis != dec2.hypothesis or dec1.decision != dec2.decision:
            return False, equal_levels
        equal_levels += 1
    return True, equal_levels

def get_random_path(tree_root: TreeNode, rng: np.random.Generator, return_random_docs: int | bool = False) -> TreePath:
    """Generate a random path from the root to a leaf in the RTP tree."""
    current_node = tree_root
    path = TreePath(decisions=[])
    
    while current_node is not None and not current_node.is_leaf():
        question = current_node.question
        if question is None:
            raise ValueError("Non-leaf node has no question.")
        
        # Randomly decide to go left or right
        if rng.random() < 0.5:
            current_node = current_node.left
            path.decisions.append(TreeDecision(hypothesis=question, decision="entailment"))
        else:
            current_node = current_node.right
            path.decisions.append(TreeDecision(hypothesis=question, decision="contradiction"))
    
    if isinstance(return_random_docs, int) and return_random_docs > 0 and current_node is not None:
        # Sample random documents from the leaf node
        sampled_docs = rng.choice(current_node.documents, size=min(return_random_docs, len(current_node.documents)), replace=False)
        sampled_docs = list(sampled_docs)
        return path, sampled_docs
    
    if current_node is None:
        raise ValueError("Reached a non-existent node in the tree.")
    
    return path
