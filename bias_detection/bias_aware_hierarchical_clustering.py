import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from scipy.stats import ttest_ind

from sklearn.cluster import k_means


class BisectKMeansWithMetric:
    def __init__(self, metric, pval=0.05, min_cluster_size=10, max_clusters_splits=50, plot=False):
        self.max_clusters_splits = max_clusters_splits
        self.min_cluster_size = min_cluster_size
        self.metric = metric
        self.pval = pval
        self.iter = 0
        self.plot = plot

    def fit_predict(self, X):
        clusters = [X]
        cluster_metric_std = [self.metric.std()]
        for i in range(self.max_clusters_splits - 1):
            cluster_with_max_std = np.argmax(cluster_metric_std)
            cluster_metric_std.pop(cluster_with_max_std)
            features_with_max_std = clusters[cluster_with_max_std]
            new_cluster_bias, new_cluster_features, new_cluster_std = self.bisect(X, features_with_max_std)

            if new_cluster_bias:
                clusters.pop(cluster_with_max_std)
                clusters += new_cluster_features
                cluster_metric_std += new_cluster_std
            else:
                clusters.pop(cluster_with_max_std)
                cluster_metric_std.append(0)
                clusters.append(features_with_max_std)
            if max(cluster_metric_std) == 0:
                break
        cluster_labels = pd.Series(np.zeros(self.metric.shape), index=self.metric.index)
        for i, clust in enumerate(clusters):
            cluster_labels[clust.index] = i
        return cluster_labels

    def bisect(self, X, features):
        for i in range(1):
            centers, labels, _ = k_means(features, 2)
            new_cluster_features = [features[labels == n] for n in range(2)]
            new_cluster_bias = self._get_significant_diff(X, new_cluster_features[0], new_cluster_features[1])
            if new_cluster_bias:
                break
        new_cluster_std = [self.metric.loc[f.index].std() if len(f) >= self.min_cluster_size else 0 for f in
                           new_cluster_features]
        return new_cluster_bias, new_cluster_features, new_cluster_std

    def _get_significant_diff(self, X, features1, features2):
        ids1, ids2 = features1.index, features2.index
        pval = ttest_ind(self.metric[ids1], self.metric[ids2]).pvalue
        self.iter += 1
        self._plot_clusters_with_metric(X, features1, features2, ids1, ids2, pval)
        return pval <= self.pval or min(len(ids1), len(ids2)) >= self.min_cluster_size

    def _plot_clusters_with_metric(self, X, features1, features2, ids1, ids2, pval: float):
        if self.plot:
            plt.figure(figsize=(12, 5))
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(wspace=5)

            sns.set(font_scale=1.3)
            sns.set_style("white")
            plt.subplot(1, 2, 1)
            plt.subplots_adjust(wspace=5)
            plt.tight_layout(pad=2.0)
            plt.subplots_adjust(wspace=5)

            ax = sns.scatterplot(x=0, y=1, data=X, s=20, alpha=.1, palette="deep")
            sns.scatterplot(x=0, y=1, data=features1, s=20, palette="deep", label='cluster1')
            sns.scatterplot(x=0, y=1, data=features2, s=20, palette="deep", label='cluster2')
            ax.set(xlabel="Feature 1", ylabel="Feature 2")
            plt.subplot(1, 2, 2)
            plt.tight_layout(pad=2.0)
            output = "p-val < 0.01" if pval < 0.01 else f"p-value={pval:.3f}"
            ax1 = sns.distplot(self.metric[ids1], hist=False, kde=True,
                               kde_kws={'linewidth': 2}, label='cluster1', color="darkorange").set_title(output)
            ax = sns.distplot(self.metric[ids2], hist=False, kde=True,
                              kde_kws={'linewidth': 2}, label='cluster2', color="green")
            ax.set(ylabel="Distribution density", xlabel="Metric value")
            plt.legend()
            plt.savefig(f'bah_km_iter{self.iter}.png', dpi=300)
            plt.show()
