import os

import pandas as pd
import numpy as np


class MovieLens100KData:

    def __init__(self, data_path: str):
        self._data_path = data_path
        self._ratings_file = os.path.join(data_path, 'u.data')
        self._item_file = os.path.join(data_path, 'u.item')
        self._user_file = os.path.join(data_path, 'u.user')
        self._genres_file = os.path.join(data_path, 'u.genre')

    def get_ratings_with_metadata(self) -> pd.DataFrame:
        user_features = self._get_user_features()
        item_features = self._get_item_features()
        ratings = self._get_ratings()
        user_activity, item_popularity = self._get_activity_features(ratings)
        ratings_with_metadata = self._join_features(ratings, user_features, item_features,
                                                    user_activity, item_popularity)
        return ratings_with_metadata

    @property
    def attributes_categorical(self):
        return ['Action',
       'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
       'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
       'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'gender']

    @property
    def attributes_continuous(self):
        return ['age', 'year', 'user activity', 'item popularity']

    def _get_ratings(self) -> pd.DataFrame:
        ratings = pd.read_csv(self._ratings_file, header=None, sep='\t')
        ratings.columns = ['user', 'item', 'rating', 'timestamp']
        min_rating = min(ratings["rating"])
        max_rating = max(ratings["rating"])
        ratings['rating_scaled'] = ratings["rating"].astype(np.float32) \
            .apply(lambda x: (x - min_rating) / (max_rating - min_rating))
        return ratings

    def _get_item_features(self):
        item_features = pd.read_csv(self._item_file, sep='|', error_bad_lines=False,
                                    encoding="latin", header=None)
        item_features.rename(columns={0: 'item', 1: 'title', 2: 'date'}, inplace=True)
        genres = pd.read_csv(self._genres_file, sep='|', header=None)
        genre_names = list(genres[0].values)
        item_features.rename(columns={i: genre_names[i - 5] for i in range(5, 24)}, inplace=True)
        item_features['year'] = pd.to_datetime(item_features['date']).apply(lambda x: x.year)
        return item_features

    def _get_user_features(self):
        users = pd.read_csv(self._user_file, sep='|', header=None)
        users.columns = ['user', 'age', 'gender', 'profession', 'zip']
        return users

    def _get_activity_features(self, ratings: pd.DataFrame):
        user_active = ratings.groupby('user').count().reset_index().rename(columns={'item': 'user activity'})
        item_popular = ratings.groupby('item').count().reset_index().rename(columns={'user': 'item popularity'})
        return user_active[['user', 'user activity']], item_popular[['item', 'item popularity']]

    def _join_features(self, ratings: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame,
                       user_activity: pd.Series, item_popularity: pd.Series):
        ratings_metadata = ratings.merge(user_features, on='user')
        ratings_metadata = ratings_metadata.merge(item_features, on='item')
        ratings_metadata = ratings_metadata.merge(user_activity, on='user').merge(item_popularity, on='item')
        ratings_metadata = ratings_metadata.merge(ratings[['user', 'item']], on=['user', 'item'])
        return ratings_metadata
