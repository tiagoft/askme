def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies unless actually needed."""
    if name in ('RTPBuilder', 'RTPRecursion'):
        from .rtp_builder import RTPBuilder, RTPRecursion
        return RTPBuilder if name == 'RTPBuilder' else RTPRecursion
    elif name in ('TreeNode', 'SplitMetrics'):
        from .tree_models import TreeNode, SplitMetrics
        return TreeNode if name == 'TreeNode' else SplitMetrics
    elif name in ('calculate_node_purity', 'calculate_all_leaf_purities', 
                  'calculate_isolation_depth', 'calculate_all_isolation_depths', 
                  'evaluate_exploratory_power'):
        from .evaluator import (
            calculate_node_purity,
            calculate_all_leaf_purities,
            calculate_isolation_depth,
            calculate_all_isolation_depths,
            evaluate_exploratory_power,
        )
        return {
            'calculate_node_purity': calculate_node_purity,
            'calculate_all_leaf_purities': calculate_all_leaf_purities,
            'calculate_isolation_depth': calculate_isolation_depth,
            'calculate_all_isolation_depths': calculate_all_isolation_depths,
            'evaluate_exploratory_power': evaluate_exploratory_power,
        }[name]
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
