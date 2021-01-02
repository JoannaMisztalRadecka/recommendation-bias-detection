import json

import pandas as pd
import seaborn as sns
from CHAID import Tree


class BiasDetectionTree:
    """
    Bias detector based on CHAID decision tree.
    Identifies combinations of attributes for which the evaluation metric is significantly different.
    """
    VAR_TYPE__CONTINUOUS = 'continuous'
    VAR_TYPE__CATEGORICAL = 'categorical'
    __NODE_RULES__COL = 'node_rules'
    __BUCKET__LOW = 'low'
    __BUCKET__MEDIUM = 'medium'
    __BUCKET__HIGH = 'high'

    def __init__(self, metric_col='error', metric_type=VAR_TYPE__CONTINUOUS,
                 min_child_node_size: int = 1000,
                 max_depth: int = 3,
                 alpha: float = 0.01):
        self.metric_col: str = metric_col
        self.metric_type = metric_type
        self.min_child_node_size: int = min_child_node_size
        self.max_depth: int = max_depth
        self.alpha: float = alpha
        self.bias_tree: Tree = None
        self.leaf_metrics: pd.DataFrame = None

    def analyze_bias(self, attributes: dict, metric_with_metadata: pd.DataFrame, plot_bias_tree: bool = True,
                     plot_nodes_distribution: bool = True, dist_type: str = 'ecdf') -> pd.Series:
        self._build_bias_tree(attributes=attributes, metric_with_metadata=metric_with_metadata)
        if plot_bias_tree:
            self._plot_bias_tree()
        leaf_metrics = self._get_nodes_metrics(metric_with_metadata=metric_with_metadata)
        if plot_nodes_distribution:
            self._plot_metric_nodes_distribution(leaf_metrics=leaf_metrics, dist_type=dist_type)
        self.leaf_metrics = leaf_metrics.groupby(self.__NODE_RULES__COL).describe()[self.metric_col].sort_values('mean')

        return self.leaf_metrics

    def get_filtered_df(self, node_rules: str, metadata_df: pd.DataFrame):
        rules_dict = json.loads(node_rules)
        rules_metadata_df = self._filter_df_rows_for_rules(rules_dict, metadata_df)
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
        tree_attributes = {}
        for attr in attributes:
            if attributes[attr] == self.VAR_TYPE__CONTINUOUS:
                var_name = f'{attr}_bucketized'
                metric_with_metadata[var_name] = pd.qcut(metric_with_metadata[attr], 3,
                                                         labels=[self.__BUCKET__LOW,
                                                                 self.__BUCKET__MEDIUM,
                                                                 self.__BUCKET__HIGH])
                tree_attributes[var_name] = 'nominal'
            elif attributes[attr] == self.VAR_TYPE__CATEGORICAL:
                tree_attributes[attr] = 'nominal'

        self.bias_tree = Tree.from_pandas_df(metric_with_metadata, tree_attributes, self.metric_col,
                                             min_child_node_size=self.min_child_node_size,
                                             dep_variable_type=self.metric_type,
                                             max_depth=self.max_depth, alpha_merge=self.alpha)

    def _plot_bias_tree(self):
        tree_structure = self.bias_tree.to_tree()
        for node in tree_structure.all_nodes():
            choices = node.tag.choices
            if node.tag.parent is not None:
                variable = self.bias_tree.tree_store[node.tag.parent].split_variable
            else:
                variable = "root"
            tree_structure.update_node(node.identifier,
                                       tag="{}={}: {}".format(variable, choices, round(node.tag.members['mean'], 2)))
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

    def _filter_df_rows_for_rules(self, node_rules: dict, metadata_df: pd.DataFrame) -> pd.DataFrame:
        rule_metadata = metadata_df.copy(deep=True)
        rule_metadata[self.__NODE_RULES__COL] = json.dumps(node_rules)
        for col in node_rules:
            rule_metadata = rule_metadata[rule_metadata[col].isin(node_rules[col])]
        return rule_metadata

    def _plot_metric_nodes_distribution(self, leaf_metrics: pd.DataFrame, dist_type: str = 'ecdf') -> None:
        sns.displot(data=leaf_metrics, x=self.metric_col, hue=self.__NODE_RULES__COL, kind=dist_type)


def residual(prediction, rating):
    return rating - prediction


def absolute_error(prediction, rating):
    return abs(rating - prediction)


def squared_error(prediction, rating):
    return (rating - prediction) ** 2


def get_metric_bias_tree_for_model(model, ratings, attributes, metric_name,
                         min_child_node_size=1000, alpha=0.01, max_depth=3):
    ratings['pred'] = model.predict(ratings[['user_id', 'movie_id']])
    ratings[metric_name] = eval(metric_name)(ratings['rating_scaled'].values, ratings['pred'].values)
    bias_detection_tree = BiasDetectionTree(min_child_node_size=min_child_node_size,
                                            alpha=alpha, max_depth=max_depth, metric_col=metric_name)
    bias_detection_tree.analyze_bias(attributes=attributes, metric_with_metadata=ratings)

    return bias_detection_tree
