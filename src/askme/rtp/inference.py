from typing import Optional
from .tree_models import TreeNode, TreeDecision, TreePath
from ..utils import NLIWithChunkingAndPooling

class TreeInference:
    """Class for performing inference on RTP trees."""

    def __init__(
        self,
        tree_root: TreeNode,
        nli_model,
        nli_tokenizer,
        device: str = 'cpu',
        chunk_size: int = 150,
        overlap: int = 50,
        ground_truth_labels: Optional[list] = None,
    ):
        self.tree_root = tree_root
        self.nli_model = nli_model
        self.nli_tokenizer = nli_tokenizer
        self.device = device
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.labels = ground_truth_labels
        self.nli_with_chunking = NLIWithChunkingAndPooling(
            nli_model=nli_model,
            batch_size=1,
            tokenizer=nli_tokenizer,
            device=device,
            chunk_size=chunk_size,
            overlap=overlap,
            disable_tqdm=True,
        )
            
    def __call__(self, document: str) -> tuple[TreeNode, TreePath, int | None]:
        """Classify a document into the RTP tree."""
        
        current_node = self.tree_root
        assert isinstance(current_node, TreeNode), "Tree root is not a TreeNode instance."
        
        path = TreePath(decisions=[])
        
        while current_node is not None and not current_node.is_leaf():
            question = current_node.question
            if question is None:
                raise ValueError("Non-leaf node has no question.")
            entails, _, _, _ = self.nli_with_chunking(
                premise=[document],
                hypothesis=question,
            )[0]
            if entails:
                current_node = current_node.left
                path.decisions.append(TreeDecision(hypothesis=question, decision="entailment"))
            else:
                current_node = current_node.right
                path.decisions.append(TreeDecision(hypothesis=question, decision="contradiction"))
        
        if current_node is None:
            raise ValueError("Reached a non-existent node in the tree.")
        
        # Select the label as the most common label in the leaf node
        if self.labels is not None:
            leaf_labels = [self.labels[idx] for idx in current_node.documents]
            from collections import Counter
            most_common_label, _ = Counter(leaf_labels).most_common(1)[0]
        else:
            most_common_label = None    
            
        return current_node, path, most_common_label
