import os

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

from data_preparation.recommender_dataset import RecommenderDataset


class MovieLens100KData(RecommenderDataset):
    """
    Prepares MovieLens dataset for recommendation model training.
    """

    NAME = 'MovieLens100K'
    USER_ID_COL = 'user'
    ITEM_ID_COL = 'item'
    RATING_COL = 'rating'
    MIN_RATING = 1
    MAX_RATING = 5

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5, bucket_labels: list=None):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins, bucket_labels=bucket_labels)
        self._ratings_file = os.path.join(data_path, 'u.data')
        self._item_file = os.path.join(data_path, 'u.item')
        self._user_file = os.path.join(data_path, 'u.user')
        self._genres_file = os.path.join(data_path, 'u.genre')

    @property
    def attributes_categorical(self):
        return ['Action',
                'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'gender']

    @property
    def attributes_continuous(self):
        return ['age', 'year', 'user activity', 'item popularity', 'date']

    def _get_ratings(self) -> pd.DataFrame:
        ratings = pd.read_csv(self._ratings_file, header=None, sep='\t')
        ratings.columns = [self.USER_ID_COL, self.ITEM_ID_COL, 'rating', 'timestamp']
        ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
        return ratings.sort_values('timestamp')

    def _get_item_features(self):
        item_features = pd.read_csv(self._item_file, sep='|', error_bad_lines=False,
                                    encoding="latin", header=None)
        item_features.rename(columns={0: self.ITEM_ID_COL, 1: 'title', 2: 'movie_date'}, inplace=True)
        genres = pd.read_csv(self._genres_file, sep='|', header=None)
        genre_names = list(genres[0].values)
        item_features.rename(columns={i: genre_names[i - 5] for i in range(5, 24)}, inplace=True)
        item_features['year'] = pd.to_datetime(item_features['movie_date']).apply(lambda x: x.year)
        return item_features

    def _get_user_features(self):
        users = pd.read_csv(self._user_file, sep='|', header=None)
        users.columns = [self.USER_ID_COL, 'age', 'gender', 'profession', 'zip']
        return users


class MovieLens1MData(RecommenderDataset):
    """
    Prepares MovieLens dataset for recommendation model training.
    """

    NAME = 'MovieLens1M'
    USER_ID_COL = 'user'
    ITEM_ID_COL = 'item'
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
        self._ratings_file = os.path.join(data_path, 'ratings.dat')
        self._item_file = os.path.join(data_path, 'movies.dat')
        self._user_file = os.path.join(data_path, 'users.dat')

    @property
    def attributes_categorical(self):
        return ['Action',
                'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'gender', 'age']

    @property
    def attributes_continuous(self):
        return ['year', 'user activity', 'item popularity', 'date']

    def _get_ratings(self) -> pd.DataFrame:
        ratings = pd.read_csv(self._ratings_file, header=None, sep='::')
        ratings.columns = [self.USER_ID_COL, self.ITEM_ID_COL, 'rating', 'timestamp']
        ratings['date'] = pd.to_datetime(ratings['timestamp'], unit='s')
        ratings.sort_values('timestamp', ascending=True, inplace=True)
        return ratings

    def _get_item_features(self):
        movie_data = pd.read_csv(self._item_file, header=None, sep='::')
        movie_data.rename(columns={0: self.ITEM_ID_COL, 1: 'title', 2: 'genres'}, inplace=True)
        vectorizer = CountVectorizer(token_pattern="[a-zA-Z0-9\-\']+", lowercase=False)
        genres_sparse = vectorizer.fit_transform(movie_data['genres'])
        movie_genre_df = pd.DataFrame.sparse.from_spmatrix(genres_sparse)
        movie_genre_df.columns = vectorizer.get_feature_names()
        movie_data = pd.concat([movie_data, movie_genre_df], 1)
        movie_data['year'] = movie_data["title"].apply(lambda x: x[-5:-1]).astype(np.int16)
        return movie_data

    def _get_user_features(self):
        users = pd.read_csv(self._user_file, sep='::', header=None)
        users.columns = [self.USER_ID_COL, 'gender', 'age', 'profession', 'zip']
        return users