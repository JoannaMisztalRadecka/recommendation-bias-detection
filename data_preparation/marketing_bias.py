import os

import pandas as pd

from data_preparation.recommender_dataset import RecommenderDataset


class MarketBiasData(RecommenderDataset):
    """
    Prepares market bias dataset for recommendation model training.
    """

    NAME = 'market bias'

    USER_ID_COL = 'user_id'
    ITEM_ID_COL = 'item_id'
    RATING_COL = 'rating'
    MIN_RATING = 1
    MAX_RATING = 5

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5, bucket_labels: list = None):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins, bucket_labels=bucket_labels)
        self._ratings_file = os.path.join(data_path, 'users_interactions.csv.zip')

    def _get_ratings(self):
        ratings = pd.read_csv(self._ratings_file).fillna('')
        ratings['date'] = pd.to_datetime(ratings['timestamp'])

        return ratings


class MarketBiasModCloth(MarketBiasData):
    NAME = 'ModCloth'

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5, bucket_labels: list = None):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins, bucket_labels=bucket_labels)
        self._ratings_file = os.path.join(data_path, 'df_modcloth.csv')

    @property
    def attributes_categorical(self):
        return ['size', 'user_attr', 'model_attr', 'category', 'brand']

    @property
    def attributes_continuous(self):
        return ['user activity', 'item popularity',  'year']


class MarketBiasElectronics(MarketBiasData):
    NAME = 'Electronics'

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5, bucket_labels: list = None):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins, bucket_labels=bucket_labels)
        self._ratings_file = os.path.join(data_path, 'df_electronics.csv')

    @property
    def attributes_categorical(self):
        return ['model_attr', 'user_attr', 'category', 'brand']

    @property
    def attributes_continuous(self):
        return ['user activity', 'item popularity', 'year']
