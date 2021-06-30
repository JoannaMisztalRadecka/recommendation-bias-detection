import pandas as pd

from bias_detection.bias_tree import BiasDetectionTree


def value(prediction, rating):
    return prediction - rating


def underestimation(prediction, rating):
    error = rating - prediction
    error[error < 0] = 0
    return error


def overestimation(prediction, rating):
    error = prediction - rating
    error[error < 0] = 0
    return error


def absolute(prediction, rating):
    return abs(rating - prediction)


def squared_error(prediction, rating):
    return (rating - prediction) ** 2


def get_metric_bias_tree_for_model(model, ratings, attributes, metric_name,
                                   min_child_node_size=1000, alpha=0.01, split_threshold=0, max_depth=3,
                                   user_col='user_id',
                                   item_col='item_id', rating_col='rating', plot_bias_tree: bool = True,
                                   dataset_name: str = ''):
    ratings['pred'] = model.predict(ratings[[user_col, item_col]])
    ratings[metric_name] = eval(metric_name)(ratings[rating_col].values, ratings['pred'].values)
    bias_detection_tree = BiasDetectionTree(min_child_node_size=min_child_node_size,
                                            alpha=alpha, max_depth=max_depth, metric_col=metric_name,
                                            dataset_name=dataset_name, split_threshold=split_threshold)
    bias_detection_tree.analyze_bias(attributes=attributes, metric_with_metadata=ratings, plot_bias_tree=plot_bias_tree,
                                     plot_nodes_distribution=plot_bias_tree)

    return bias_detection_tree


def evaluate_model(model, ratings: pd.DataFrame, metric_name: str, user_col='user_id',
                   item_col='item_id', rating_col='rating') -> pd.Series:
    ratings['pred'] = model.predict(ratings[[user_col, item_col]])
    metric = eval(metric_name)(ratings[rating_col], ratings['pred'])
    return metric