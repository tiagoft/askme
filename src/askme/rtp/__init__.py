def __getattr__(name):
    """Lazy import to avoid loading heavy dependencies unless actually needed."""
    if name in ('RTPBuilder', 'RTPRecursion'):
        from .rtp_builder import RTPBuilder, RTPRecursion
        return RTPBuilder if name == 'RTPBuilder' else RTPRecursion
    elif name in ('KMeansTreeBuilder', 'KMeansTreeRecursion'):
        from .kmeans_tree_builder import KMeansTreeBuilder, KMeansTreeRecursion
        return KMeansTreeBuilder if name == 'KMeansTreeBuilder' else KMeansTreeRecursion
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
        from .metrics.self_supervised_metrics import (
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
    # Supervised metrics
    elif name == 'SupervisedMetric':
        from .metrics.supervised_metrics import SupervisedMetric
        return SupervisedMetric
    elif name == 'NormalizedMutualInformation':
        from .metrics.supervised_metrics import NormalizedMutualInformation
        return NormalizedMutualInformation
    elif name == 'AdjustedRandIndex':
        from .metrics.supervised_metrics import AdjustedRandIndex
        return AdjustedRandIndex
    elif name == 'HomogeneityCompletenessVMeasure':
        from .metrics.supervised_metrics import HomogeneityCompletenessVMeasure
        return HomogeneityCompletenessVMeasure
    elif name == 'Accuracy':
        from .metrics.supervised_metrics import Accuracy
        return Accuracy
    elif name == 'F1Score':
        from .metrics.supervised_metrics import F1Score
        return F1Score
    elif name == 'ConfusionMatrix':
        from .metrics.supervised_metrics import ConfusionMatrix
        return ConfusionMatrix
    elif name == 'UnsupervisedMetric':
        from .metrics.unsupervised_metrics import UnsupervisedMetric
        return UnsupervisedMetric
    elif name == 'NumberOfNodes':
        from .metrics.unsupervised_metrics import NumberOfNodes
        return NumberOfNodes
    elif name == 'TreeHeight':
        from .metrics.unsupervised_metrics import TreeHeight
        return TreeHeight
    elif name == 'NumberOfLeafNodes':
        from .metrics.unsupervised_metrics import NumberOfLeafNodes
        return NumberOfLeafNodes
    elif name == 'TreeNodeUnbalance':
        from .metrics.unsupervised_metrics import TreeNodeUnbalance
        return TreeNodeUnbalance
    elif name == 'DocumentsPerLeaf':
        from .metrics.unsupervised_metrics import DocumentsPerLeaf
        return DocumentsPerLeaf
    elif name == 'TreeDocumentUnbalance':
        from .metrics.unsupervised_metrics import TreeDocumentUnbalance
        return TreeDocumentUnbalance
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__all__ = [
    'RTPBuilder',
    'RTPRecursion',
    'KMeansTreeBuilder',
    'KMeansTreeRecursion',
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
    'SupervisedMetric',
    'NormalizedMutualInformation',
    'AdjustedRandIndex',
    'HomogeneityCompletenessVMeasure',
    'Accuracy',
    'F1Score',
    'ConfusionMatrix',
    'UnsupervisedMetric',
    'NumberOfNodes',
    'TreeHeight',
    'NumberOfLeafNodes',
    'TreeNodeUnbalance',
    'DocumentsPerLeaf',
    'TreeDocumentUnbalance',
]
