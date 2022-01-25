from abc import abstractmethod, abstractproperty, ABC

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor


class RecommenderDataset(ABC):
    """
    Prepares dataset for recommendation model training.
    """
    NAME = 'Recommendation Dataset'
    USER_ID_COL = 'user'
    ITEM_ID_COL = 'item'
    USER_ID_COL_TRANSFORMED = 'user_id'
    ITEM_ID_COL_TRANSFORMED = 'item_id'
    RATING_COL = 'rating'
    MIN_RATING = 1
    MAX_RATING = 5

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 4, bucket_labels: list = None, min_feature_cnt: int = 500):
        """
        :param data_path: Path with dataset.
        """
        self._data_path = data_path
        self._user_ids = []
        self._item_ids = []
        self._min_user_interactions = min_user_interactions
        self._min_item_interactions = min_item_interactions
        self._min_feature_cnt = min_feature_cnt
        self._sample = sample
        self._n_bins = n_bins
        self._bucket_labels = bucket_labels

    def get_data_splits_for_training(self, use_val_set: bool = True, use_timestamp_col: str = None,
                                     shuffle: bool = True) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Prepares MovieLens data for training.
        :param ratings: pandas DataFrame with ratings
        :return: train, validation, test sets (60/20/20)
        """
        ratings = self.get_ratings_with_metadata()
        self._user_ids = ratings[self.USER_ID_COL].unique().tolist()
        user_ids = {x: i for i, x in enumerate(self._user_ids)}
        self._item_ids = ratings[self.ITEM_ID_COL].unique().tolist()
        item_ids = {x: i for i, x in enumerate(self._item_ids)}
        ratings[self.USER_ID_COL_TRANSFORMED] = ratings[self.USER_ID_COL].map(user_ids)
        ratings[self.ITEM_ID_COL_TRANSFORMED] = ratings[self.ITEM_ID_COL].map(item_ids)
        if use_timestamp_col is not None:
            ratings_sorted = ratings.sort_values(use_timestamp_col)
            train_size = round(.8 * ratings_sorted.shape[0])
            return ratings_sorted[:train_size], ratings_sorted[train_size:]

        return self._get_train_test_validation_data(ratings, use_val_set, shuffle=shuffle)

    def get_ratings_with_metadata(self) -> pd.DataFrame:
        user_features = self._get_user_features()
        item_features = self._get_item_features()
        ratings = self._get_ratings()
        user_activity, item_popularity = self._get_activity_features(ratings)
        ratings_with_metadata = self._join_features(ratings, user_features, item_features,
                                                    user_activity, item_popularity, )
        ratings_with_metadata = self._bucketize_continuous_attributes(ratings_with_metadata, self._bucket_labels)
        if self._sample:
            ratings_with_metadata = ratings_with_metadata.sample(self._sample)
        for attr in self.attributes_dict:
            popular_attr = ratings_with_metadata[attr].value_counts()[
                ratings_with_metadata[attr].value_counts() > self._min_feature_cnt].index
            ratings_with_metadata[attr] = ratings_with_metadata[attr].apply(
                lambda x: x if x in popular_attr else attr + ':unpopular')

        return ratings_with_metadata

    @property
    def user_ids(self):
        return self._user_ids

    @property
    def item_ids(self):
        return self._item_ids

    @property
    def attributes_dict(self):
        attributes = {f'{attr}_bucketized': 'nominal' for attr in self.attributes_continuous}
        attributes.update({attr: 'nominal' for attr in self.attributes_categorical})
        return attributes

    @abstractproperty
    def attributes_categorical(self) -> list:
        return []

    @abstractproperty
    def attributes_continuous(self) -> list:
        return []

    @abstractmethod
    def _get_ratings(self) -> pd.DataFrame:
        pass

    def _get_item_features(self) -> pd.DataFrame:
        return None

    def _get_user_features(self) -> pd.DataFrame:
        return None

    def _bucketize_continuous_attributes(self, ratings_with_metadata: pd.DataFrame,
                                         bucket_labels: list = None) -> pd.DataFrame:
        for attr in self.attributes_continuous:
            var_name = f'{attr}_bucketized'
            ratings_with_metadata[var_name] = pd.qcut(ratings_with_metadata[attr], self._n_bins,
                                                      duplicates='drop', labels=bucket_labels, precision=0).astype(str)

        return ratings_with_metadata

    def _get_activity_features(self, ratings: pd.DataFrame):
        user_active = ratings.groupby(self.USER_ID_COL).count().reset_index().rename(
            columns={self.ITEM_ID_COL: 'user activity'})
        user_active = user_active[user_active['user activity'] >= self._min_user_interactions]
        item_popular = ratings.groupby(self.ITEM_ID_COL).count().reset_index().rename(
            columns={self.USER_ID_COL: 'item popularity'})
        item_popular = item_popular[item_popular['item popularity'] >= self._min_item_interactions]
        return user_active[[self.USER_ID_COL, 'user activity']], item_popular[[self.ITEM_ID_COL, 'item popularity']]

    def _get_lof_features(self, ratings: pd.DataFrame):
        rating_mx = ratings.pivot_table(index=self.USER_ID_COL, columns=self.ITEM_ID_COL,
                                        values=self.RATING_COL).fillna(0)
        clf = LocalOutlierFactor(n_neighbors=30, metric='euclidean')
        clf.fit(rating_mx)
        user_lof = pd.DataFrame(clf.negative_outlier_factor_)
        user_lof.columns = ['LOF']
        user_lof[self.USER_ID_COL] = rating_mx.index
        return user_lof

    def _join_features(self, ratings: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame,
                       user_activity: pd.Series, item_popularity: pd.Series, ):
        ratings_metadata = ratings.copy(deep=True)
        if user_features is not None:
            ratings_metadata = ratings.merge(user_features,
                                             on=self.USER_ID_COL)
        if item_features is not None:
            ratings_metadata = ratings_metadata.merge(item_features, on=self.ITEM_ID_COL)
        if user_activity is not None:
            ratings_metadata = ratings_metadata.merge(user_activity, on=self.USER_ID_COL)
        if item_popularity is not None:
                ratings_metadata = ratings_metadata.merge(item_popularity, on=self.ITEM_ID_COL)

        return ratings_metadata

    @staticmethod
    def _get_train_test_validation_data(ratings: pd.DataFrame, use_val_set: bool = True, shuffle: bool = True) -> (
            pd.DataFrame, pd.DataFrame, pd.DataFrame):
        X_train, X_test = train_test_split(ratings, test_size=0.2, random_state=1, shuffle=shuffle)
        if use_val_set:
            X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1, shuffle=shuffle)
            return X_train, X_val, X_test
        else:
            return X_train, X_test
