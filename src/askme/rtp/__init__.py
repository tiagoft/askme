def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies unless actually needed."""
    if name in ('RTPBuilder', 'RTPRecursion'):
        from .rtp_builder import RTPBuilder, RTPRecursion
        return RTPBuilder if name == 'RTPBuilder' else RTPRecursion
    elif name in ('TreeNode', 'SplitMetrics'):
        from .tree_models import TreeNode, SplitMetrics
        return TreeNode if name == 'TreeNode' else SplitMetrics
    elif name == 'calculate_node_purity':
        from .evaluator import calculate_node_purity
        return calculate_node_purity
    elif name == 'calculate_all_leaf_purities':
        from .evaluator import calculate_all_leaf_purities
        return calculate_all_leaf_purities
    elif name == 'calculate_isolation_depth':
        from .evaluator import calculate_isolation_depth
        return calculate_isolation_depth
    elif name == 'calculate_all_isolation_depths':
        from .evaluator import calculate_all_isolation_depths
        return calculate_all_isolation_depths
    elif name == 'evaluate_exploratory_power':
        from .evaluator import evaluate_exploratory_power
        return evaluate_exploratory_power
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'RTPBuilder',
    'RTPRecursion',
    'TreeNode',
    'SplitMetrics',
    'calculate_node_purity',
    'calculate_all_leaf_purities',
    'calculate_isolation_depth',
    'calculate_all_isolation_depths',
    'evaluate_exploratory_power',
]
