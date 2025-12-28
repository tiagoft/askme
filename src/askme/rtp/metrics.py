"""
Metrics module for calculating entropy and information gain in RTP trees.

This module provides functions to evaluate the quality of splits in 
Recursive Thematic Partitioning based on information theory metrics.
"""

import math
from typing import List, Union


def calculate_entropy(labels: List[Union[int, str]]) -> float:
    """
    Calculate Shannon Entropy H(D) for a set of labels.
    
    Entropy measures the impurity or uncertainty in a dataset.
    H(D) = -Σ(p_i * log2(p_i)) where p_i is the proportion of class i.
    
    Args:
        labels: List of ground-truth labels (can be integers or strings)
        
    Returns:
        float: Shannon entropy value. Returns 0.0 for empty or homogeneous sets.
        
    Example:
        >>> calculate_entropy([0, 0, 0, 0])  # Pure set
        0.0
        >>> calculate_entropy([0, 1, 0, 1])  # Maximum entropy for binary
        1.0
    """
    if not labels:
        return 0.0
    
    # Count occurrences of each label
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    total = len(labels)
    entropy = 0.0
    
    for count in label_counts.values():
        if count > 0:
            probability = count / total
            entropy -= probability * math.log2(probability)
    
    return entropy


def calculate_information_gain(
    parent_labels: List[Union[int, str]],
    left_labels: List[Union[int, str]],
    right_labels: List[Union[int, str]]
) -> float:
    """
    Calculate Information Gain (IG) for a split.
    
    Information Gain measures the reduction in entropy achieved by splitting
    the parent node into left and right child nodes.
    
    IG(D, split) = H(D) - [|D_left|/|D| * H(D_left) + |D_right|/|D| * H(D_right)]
    
    Args:
        parent_labels: Labels in the parent node before split
        left_labels: Labels in the left child node after split
        right_labels: Labels in the right child node after split
        
    Returns:
        float: Information gain value (non-negative)
        
    Example:
        >>> parent = [0, 0, 1, 1]
        >>> left = [0, 0]
        >>> right = [1, 1]
        >>> calculate_information_gain(parent, left, right)
        1.0  # Perfect split
    """
    if not parent_labels:
        return 0.0
    
    parent_entropy = calculate_entropy(parent_labels)
    
    total = len(parent_labels)
    left_weight = len(left_labels) / total
    right_weight = len(right_labels) / total
    
    left_entropy = calculate_entropy(left_labels)
    right_entropy = calculate_entropy(right_labels)
    
    # Weighted average of child entropies
    weighted_child_entropy = (left_weight * left_entropy + 
                              right_weight * right_entropy)
    
    information_gain = parent_entropy - weighted_child_entropy
    
    return information_gain
