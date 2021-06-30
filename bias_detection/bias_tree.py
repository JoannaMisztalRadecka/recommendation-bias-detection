import json

import pandas as pd
import seaborn as sns
from CHAID import Tree
from matplotlib import pyplot as plt


class BiasDetectionTree:
    """
    Bias detector based on CHAID decision tree.
    Identifies combinations of attributes for which the evaluation metric is significantly different.
    """
    VAR_TYPE__CONTINUOUS = 'continuous'
    VAR_TYPE__CATEGORICAL = 'categorical'
    __NODE_RULES__COL = 'node_rules'

    def __init__(self, metric_col='error', metric_type=VAR_TYPE__CONTINUOUS,
                 min_child_node_size: int = 1000,
                 max_depth: int = 3,
                 alpha: float = 0.01, split_threshold=0, dataset_name: str = ''):
        self.metric_col: str = metric_col
        self.metric_type = metric_type
        self.min_child_node_size: int = min_child_node_size
        self.max_depth: int = max_depth
        self.alpha: float = alpha
        self.split_threshold = split_threshold
        self.bias_tree: Tree = None
        self.leaf_metrics: pd.DataFrame = None
        self.dataset_name = dataset_name

    def analyze_bias(self, attributes: dict, metric_with_metadata: pd.DataFrame, plot_bias_tree: bool = True,
                     plot_nodes_distribution: bool = True) -> pd.DataFrame:

        self._build_bias_tree(attributes=attributes, metric_with_metadata=metric_with_metadata)
        if plot_bias_tree:
            self._plot_bias_tree()
        leaf_metrics = self._get_nodes_metrics(metric_with_metadata=metric_with_metadata)
        self.leaf_metrics = leaf_metrics.groupby(self.__NODE_RULES__COL)[self.metric_col].describe().sort_values(
            'mean')

        self.leaf_metrics['global'] = leaf_metrics[self.metric_col].mean()
        if plot_nodes_distribution:
            self._plot_metric_nodes_distribution(leaf_metrics=leaf_metrics)
        return self.leaf_metrics

    @staticmethod
    def get_filtered_df(node_rules: str, metadata_df: pd.DataFrame):
        rules_dict = json.loads(node_rules)
        rules_metadata_df = BiasDetectionTree._filter_df_rows_for_rules(rules_dict, metadata_df)
        return rules_metadata_df

    @property
    def max_metric_node(self):
        return self.leaf_metrics.iloc[-1].name

    @property
    def max_metric_value(self):
        return self.leaf_metrics.iloc[-1]['mean']

    @property
    def min_metric_node(self):
        return self.leaf_metrics.iloc[0].name

    @property
    def min_metric_value(self):
        return self.leaf_metrics.iloc[0]['mean']

    def _build_bias_tree(self, attributes: dict, metric_with_metadata: pd.DataFrame) -> None:
        self.bias_tree = Tree.from_pandas_df(metric_with_metadata, attributes, self.metric_col,
                                             min_child_node_size=self.min_child_node_size,
                                             dep_variable_type=self.metric_type,
                                             max_depth=self.max_depth, alpha_merge=self.alpha,
                                             split_threshold=self.split_threshold)

    def _plot_bias_tree(self):
        tree_structure = self.bias_tree.to_tree()
        for node in tree_structure.all_nodes():
            choices = node.tag.choices
            if node.tag.parent is not None:
                variable = self.bias_tree.tree_store[node.tag.parent].split_variable
            else:
                variable = "root"
            tree_structure.update_node(node.identifier,
                                       tag="{}={}: {}".format(variable, choices, round(node.tag.members['mean'], 3)))
        tree_structure.show()

    def _get_nodes_metrics(self, metric_with_metadata: pd.DataFrame) -> pd.DataFrame:
        tree_structure = self.bias_tree.to_tree()
        node_rows = []
        for path_to_leaf in tree_structure.paths_to_leaves():
            node_rules = self._get_node_rules(tree_structure, path_to_leaf)
            rule_metadata = self._filter_df_rows_for_rules(node_rules, metric_with_metadata)
            node_rows.append(rule_metadata)
            del rule_metadata
        return pd.concat(node_rows)

    def _get_node_rules(self, tree_structure, path_to_leaf) -> dict:
        node_rules = {}
        for node_id in path_to_leaf[1:]:
            node = tree_structure.get_node(node_id).tag
            choices = node.choices
            variable = self.bias_tree.tree_store[node.parent].split_variable
            node_rules[variable] = choices
        return node_rules

    @staticmethod
    def _filter_df_rows_for_rules(node_rules: dict, metadata_df: pd.DataFrame) -> pd.DataFrame:
        rule_metadata = metadata_df.copy(deep=True)
        rule_metadata[BiasDetectionTree.__NODE_RULES__COL] = json.dumps(node_rules)
        for col in node_rules:
            rule_metadata = rule_metadata[rule_metadata[col].isin(node_rules[col])]
        return rule_metadata

    def _plot_metric_nodes_distribution(self, leaf_metrics: pd.DataFrame, dist_type: str = 'kde') -> None:
        g = sns.displot(data=leaf_metrics, x=self.metric_col, hue=self.__NODE_RULES__COL, kind=dist_type,
                        common_norm=False)
        g.savefig(f"bias_distribution_{self.metric_col}-{self.dataset_name}.png", dpi=600)
        plt.show()


