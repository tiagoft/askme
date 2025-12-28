from pydantic import BaseModel
from typing import Optional

class TreeNode(BaseModel):
    documents: list[int]  # Indices of documents in this node
    left: Optional['TreeNode'] = None  # Left child node
    right: Optional['TreeNode'] = None  # Right child node
    #parent: Optional['TreeNode'] = None  # Parent node
    question: Optional[str] = None  # Question used to split this node


TreeNode.model_rebuild()

class TokenUsage(BaseModel):
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0