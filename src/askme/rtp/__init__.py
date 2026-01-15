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
    elif name == 'calculate_node_entropy':
        from .evaluator import calculate_node_entropy
        return calculate_node_entropy
    elif name == 'calculate_all_leaf_purities':
        from .evaluator import calculate_all_leaf_purities
        return calculate_all_leaf_purities
    elif name == 'calculate_all_leaf_entropies':
        from .evaluator import calculate_all_leaf_entropies
        return calculate_all_leaf_entropies
    elif name == 'calculate_isolation_depth':
        from .evaluator import calculate_isolation_depth
        return calculate_isolation_depth
    elif name == 'calculate_all_isolation_depths':
        from .evaluator import calculate_all_isolation_depths
        return calculate_all_isolation_depths
    elif name == 'evaluate_exploratory_power':
        from .evaluator import evaluate_exploratory_power
        return evaluate_exploratory_power
    elif name == 'run_hdbscan_baseline':
        from .hdbscan_baseline import run_hdbscan_baseline
        return run_hdbscan_baseline
    elif name == 'calculate_tree_depth':
        from .hdbscan_baseline import calculate_tree_depth
        return calculate_tree_depth
    elif name == 'query':
        from .query import query
        return query
    elif name == 'run_bertopic_baseline':
        from .bertopic_baseline import run_bertopic_baseline
        return run_bertopic_baseline
    elif name == 'load_tree_from_json':
        from .tree_to_pdf import load_tree_from_json
        return load_tree_from_json
    elif name == 'tree_to_graphviz':
        from .tree_to_pdf import tree_to_graphviz
        return tree_to_graphviz
    elif name == 'tree_to_pdf':
        from .tree_to_pdf import tree_to_pdf
        return tree_to_pdf
    elif name in ('SelfSupervisedMetric', 'SilhouetteScoreMetric', 'DaviesBouldinScoreMetric',
                  'CalinskiHarabaszScoreMetric', 'TopicDiversityMetric', 'ChildParentUniquenessMetric'):
        from .self_supervised_metrics import (
            SelfSupervisedMetric,
            SilhouetteScoreMetric,
            DaviesBouldinScoreMetric,
            CalinskiHarabaszScoreMetric,
            TopicDiversityMetric,
            ChildParentUniquenessMetric,
        )
        return {
            'SelfSupervisedMetric': SelfSupervisedMetric,
            'SilhouetteScoreMetric': SilhouetteScoreMetric,
            'DaviesBouldinScoreMetric': DaviesBouldinScoreMetric,
            'CalinskiHarabaszScoreMetric': CalinskiHarabaszScoreMetric,
            'TopicDiversityMetric': TopicDiversityMetric,
            'ChildParentUniquenessMetric': ChildParentUniquenessMetric,
        }[name]
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'RTPBuilder',
    'RTPRecursion',
    'TreeNode',
    'SplitMetrics',
    'calculate_node_purity',
    'calculate_node_entropy',
    'calculate_all_leaf_purities',
    'calculate_all_leaf_entropies',
    'calculate_isolation_depth',
    'calculate_all_isolation_depths',
    'evaluate_exploratory_power',
    'run_hdbscan_baseline',
    'run_bertopic_baseline',
    'calculate_tree_depth',
    'query',
    'load_tree_from_json',
    'tree_to_graphviz',
    'tree_to_pdf',
    'SelfSupervisedMetric',
    'SilhouetteScoreMetric',
    'DaviesBouldinScoreMetric',
    'CalinskiHarabaszScoreMetric',
    'TopicDiversityMetric',
    'ChildParentUniquenessMetric',
]
