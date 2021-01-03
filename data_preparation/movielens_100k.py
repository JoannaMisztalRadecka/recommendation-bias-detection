import os

import pandas as pd
from sklearn.model_selection import train_test_split


class MovieLens100KData:
    """
    Prepares MovieLens dataset for recommendation model training.
    """
    __BUCKET__LOW = 'low'
    __BUCKET__MEDIUM = 'medium'
    __BUCKET__HIGH = 'high'

    def __init__(self, data_path: str):
        """
        :param data_path: Path with MovieLens dataset.
        """
        self._data_path = data_path
        self._ratings_file = os.path.join(data_path, 'u.data')
        self._item_file = os.path.join(data_path, 'u.item')
        self._user_file = os.path.join(data_path, 'u.user')
        self._genres_file = os.path.join(data_path, 'u.genre')
        self._num_users = 0
        self._num_items = 0

    def get_data_splits_for_training(self, use_val_set: bool = True) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """
        Prepares MovieLens data for training.
        :param ratings: pandas DataFrame with ratings
        :return: train, validation, test sets (60/20/20)
        """
        ratings = self.get_ratings_with_metadata()
        user_ids = ratings["user"].unique().tolist()
        user_ids = {x: i for i, x in enumerate(user_ids)}
        item_ids = ratings["item"].unique().tolist()
        item_ids = {x: i for i, x in enumerate(item_ids)}
        ratings["user_id"] = ratings["user"].map(user_ids)
        ratings["item_id"] = ratings["item"].map(item_ids)
        self._num_users = len(user_ids)
        self._num_items = len(item_ids)
        return self._get_train_test_validation_data(ratings, use_val_set)

    def get_ratings_with_metadata(self) -> pd.DataFrame:
        user_features = self._get_user_features()
        item_features = self._get_item_features()
        ratings = self._get_ratings()
        user_activity, item_popularity = self._get_activity_features(ratings)
        ratings_with_metadata = self._join_features(ratings, user_features, item_features,
                                                    user_activity, item_popularity)
        ratings_with_metadata = self._bucketize_continuous_attributes(ratings_with_metadata)

        return ratings_with_metadata

    @property
    def num_users(self):
        return self._num_users

    @property
    def num_items(self):
        return self._num_items

    @property
    def attributes_dict(self):
        attributes = {f'{attr}_bucketized': 'nominal' for attr in self.attributes_continuous}
        attributes.update({attr: 'nominal' for attr in self.attributes_categorical})
        return attributes

    @property
    def attributes_categorical(self):
        return ['Action',
                'Adventure', 'Animation', "Children's", 'Comedy', 'Crime',
                'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical',
                'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western', 'gender']

    @property
    def attributes_continuous(self):
        return ['age', 'year', 'user activity', 'item popularity']

    def _bucketize_continuous_attributes(self, ratings_with_metadata: pd.DataFrame) -> pd.DataFrame:
        for attr in self.attributes_continuous:
            var_name = f'{attr}_bucketized'
            ratings_with_metadata[var_name] = pd.qcut(ratings_with_metadata[attr], 3,
                                                      labels=[self.__BUCKET__LOW,
                                                              self.__BUCKET__MEDIUM,
                                                              self.__BUCKET__HIGH])
        return ratings_with_metadata

    @staticmethod
    def _get_train_test_validation_data(ratings: pd.DataFrame, use_val_set: bool = True) -> (
    pd.DataFrame, pd.DataFrame, pd.DataFrame):
        X_train, X_test = train_test_split(ratings, test_size=0.2, random_state=1)
        if use_val_set:
            X_train, X_val = train_test_split(X_train, test_size=0.25, random_state=1)
            return X_train, X_val, X_test
        else:
            return X_train, X_test

    def _get_ratings(self) -> pd.DataFrame:
        ratings = pd.read_csv(self._ratings_file, header=None, sep='\t')
        ratings.columns = ['user', 'item', 'rating', 'timestamp']

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

    @staticmethod
    def _get_activity_features(ratings: pd.DataFrame):
        user_active = ratings.groupby('user').count().reset_index().rename(columns={'item': 'user activity'})
        item_popular = ratings.groupby('item').count().reset_index().rename(columns={'user': 'item popularity'})
        return user_active[['user', 'user activity']], item_popular[['item', 'item popularity']]

    @staticmethod
    def _join_features(ratings: pd.DataFrame, user_features: pd.DataFrame, item_features: pd.DataFrame,
                       user_activity: pd.Series, item_popularity: pd.Series):
        ratings_metadata = ratings.merge(user_features, on='user')
        ratings_metadata = ratings_metadata.merge(item_features, on='item')
        ratings_metadata = ratings_metadata.merge(user_activity, on='user').merge(item_popularity, on='item')
        ratings_metadata = ratings_metadata.merge(ratings[['user', 'item']], on=['user', 'item'])
        return ratings_metadata
