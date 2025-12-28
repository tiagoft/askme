"""
Tests for entropy and information gain metrics.
"""

import pytest
import math
from askme.rtp.metrics import calculate_entropy, calculate_information_gain


def test_entropy_empty_list():
    """Entropy of empty set should be 0."""
    assert calculate_entropy([]) == 0.0


def test_entropy_pure_set():
    """Entropy of a pure (homogeneous) set should be 0."""
    assert calculate_entropy([0, 0, 0, 0]) == 0.0
    assert calculate_entropy(['A', 'A', 'A']) == 0.0


def test_entropy_binary_equal_split():
    """Entropy of evenly split binary set should be 1.0."""
    entropy = calculate_entropy([0, 1, 0, 1])
    assert abs(entropy - 1.0) < 1e-10


def test_entropy_binary_unequal_split():
    """Entropy of unevenly split binary set should be between 0 and 1."""
    entropy = calculate_entropy([0, 0, 0, 1])
    expected = -(0.75 * math.log2(0.75) + 0.25 * math.log2(0.25))
    assert abs(entropy - expected) < 1e-10


def test_entropy_three_classes():
    """Entropy of three-class set with equal distribution."""
    # Three classes, equal distribution: H = log2(3) ≈ 1.585
    entropy = calculate_entropy([0, 1, 2, 0, 1, 2, 0, 1, 2])
    expected = math.log2(3)
    assert abs(entropy - expected) < 1e-10


def test_entropy_with_strings():
    """Entropy should work with string labels."""
    entropy = calculate_entropy(['cat', 'dog', 'cat', 'dog'])
    assert abs(entropy - 1.0) < 1e-10


def test_information_gain_empty_parent():
    """IG with empty parent should be 0."""
    ig = calculate_information_gain([], [], [])
    assert ig == 0.0


def test_information_gain_perfect_split():
    """IG for a perfect split should equal parent entropy."""
    parent = [0, 0, 1, 1]
    left = [0, 0]
    right = [1, 1]
    ig = calculate_information_gain(parent, left, right)
    parent_entropy = calculate_entropy(parent)
    assert abs(ig - parent_entropy) < 1e-10
    assert abs(ig - 1.0) < 1e-10


def test_information_gain_no_split():
    """IG should be 0 when split doesn't separate classes."""
    parent = [0, 1, 0, 1]
    left = [0, 1]
    right = [0, 1]
    ig = calculate_information_gain(parent, left, right)
    assert abs(ig - 0.0) < 1e-10


def test_information_gain_unbalanced_split():
    """IG for an unbalanced but good split."""
    parent = [0, 0, 0, 1, 1, 1]
    left = [0, 0, 0]
    right = [1, 1, 1]
    ig = calculate_information_gain(parent, left, right)
    # Perfect split, IG = parent entropy = 1.0
    assert abs(ig - 1.0) < 1e-10


def test_information_gain_partial_improvement():
    """IG for a split that partially improves purity."""
    parent = [0, 0, 1, 1, 1, 1]
    left = [0, 0, 1]
    right = [1, 1, 1]
    ig = calculate_information_gain(parent, left, right)
    # IG should be positive but less than parent entropy
    parent_entropy = calculate_entropy(parent)
    assert 0 < ig < parent_entropy


def test_information_gain_with_strings():
    """IG should work with string labels."""
    parent = ['cat', 'cat', 'dog', 'dog']
    left = ['cat', 'cat']
    right = ['dog', 'dog']
    ig = calculate_information_gain(parent, left, right)
    assert abs(ig - 1.0) < 1e-10
