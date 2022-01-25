import os
import pandas as pd
from data_preparation.recommender_dataset import RecommenderDataset


class HomeAppliancesData(RecommenderDataset):
    """
    Prepares home home_appliances bias dataset for recommendation model training.
    """

    NAME = 'home appliances bias'

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5, bucket_labels: list = None):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins, bucket_labels=bucket_labels)
        self._ratings_file = os.path.join(data_path, 'data_for_bias_detection.csv')

    def _get_ratings(self):
        ratings = pd.read_csv(self._ratings_file, sep=';')
        ratings['metric'] = ratings['metric'].astype(float)
        ratings['chosen_favourite'] = (ratings['chosen_brand'] == ratings['favourite_brand']).astype(int)
        return ratings

    def _get_activity_features(self, ratings: pd.DataFrame):
        return None, None

    @property
    def attributes_categorical(self):
        return ['age_group', 'chosen_favourite', 'favourite_brand', 'chosen_brand']

    @property
    def attributes_continuous(self):
        return []
