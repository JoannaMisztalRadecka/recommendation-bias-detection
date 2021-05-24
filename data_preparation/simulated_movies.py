import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from data_preparation.recommender_dataset import RecommenderDataset


class SimulatedMovieData(RecommenderDataset):
    """
    Prepares simulated dataset for recommendation model training.
    """

    NAME = 'Simulated movie data'
    USER_ID_COL = 'user'
    ITEM_ID_COL = 'item'
    RATING_COL = 'rating'
    MIN_RATING = 1
    MAX_RATING = 5

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5, bucket_labels: list=None,
                        n_interactions=10000):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins, bucket_labels=bucket_labels)
        self.n_users = 500
        self.n_items = 500
        self.n_interactions = n_interactions

    def get_ratings_with_metadata(self):
        ratings = self._get_ratings()
        genres = ['Action', 'Comedy', 'Crime', 'Thriller']
        ratings['genre'] = np.random.choice(genres, [self.n_interactions,])
        ratings['gender'] = np.random.choice(["Female", "Male", "Other"], [self.n_interactions,])
        ratings["age"] =  pd.Series(np.random.normal(40, 15, size=[self.n_interactions,])).astype(int)
        ratings["year"] =  pd.Series(np.random.normal(2000, 15, size=[self.n_interactions,])).astype(int)
        ratings_with_metadata = self._bucketize_continuous_attributes(ratings, self._bucket_labels)
        return ratings_with_metadata

    @property
    def attributes_categorical(self):
        return ['genre', 'gender']

    @property
    def attributes_continuous(self):
        return ['age', 'year']

    def _get_ratings(self) -> pd.DataFrame:
        ratings = pd.DataFrame(np.random.choice(self.n_items, size=[self.n_interactions, 2]))
        ratings.columns = ['user', 'item']
        ratings['rating'] = pd.Series(np.random.choice(5, size=[self.n_interactions,]))
        return ratings


class SimulatedGenericData(RecommenderDataset):
    """
    Prepares simulated dataset for recommendation model training.
    """

    NAME = 'Simulated data'
    USER_ID_COL = 'user'
    ITEM_ID_COL = 'item'
    RATING_COL = 'rating'
    MIN_RATING = 1
    MAX_RATING = 5

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5, bucket_labels: list=None,  n_interactions=10000,
                        n_attributes=5, n_categories_per_attribute=4):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins, bucket_labels=bucket_labels)
        self.n_users = 500
        self.n_items = 500
        self.n_interactions = n_interactions
        self.n_attributes = n_attributes
        self.n_categories_per_attribute = n_categories_per_attribute
        self.categories = [f'c{i}' for i in range(self.n_categories_per_attribute)]
        self.attributes = [f'a{i}' for i in range(self.n_attributes)]

    def get_ratings_with_metadata(self):
        ratings = self._get_ratings()
        for a in self.attributes:
            ratings[a] = np.random.choice(self.categories, [self.n_interactions,])
        return ratings

    @property
    def attributes_categorical(self):
        return self.attributes

    @property
    def attributes_continuous(self):
        return []

    def _get_ratings(self) -> pd.DataFrame:
        ratings = pd.DataFrame(np.random.choice(self.n_items, size=[self.n_interactions, 2]))
        ratings.columns = ['user', 'item']
        ratings['rating'] = pd.Series(np.random.choice(5, size=[self.n_interactions,]))
        return ratings


        