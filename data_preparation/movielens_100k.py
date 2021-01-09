import os

import pandas as pd

from data_preparation.recommender_dataset import RecommenderDataset


class MovieLens100KData(RecommenderDataset):
    """
    Prepares MovieLens dataset for recommendation model training.
    """
    __BUCKET__LOW = 'low'
    __BUCKET__MEDIUM = 'medium'
    __BUCKET__HIGH = 'high'
    NAME = 'MovieLens100K'
    USER_ID_COL = 'user'
    ITEM_ID_COL = 'item'
    RATING_COL = 'rating'
    SENSITIVE_ATTRIBUTES = ['country', 'age_group']
    MIN_RATING = 1
    MAX_RATING = 5

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins)
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
        return ['age', 'year', 'user activity', 'item popularity', ]

    def _get_ratings(self) -> pd.DataFrame:
        ratings = pd.read_csv(self._ratings_file, header=None, sep='\t')
        ratings.columns = [self.USER_ID_COL, self.ITEM_ID_COL, 'rating', 'timestamp']

        return ratings

    def _get_item_features(self):
        item_features = pd.read_csv(self._item_file, sep='|', error_bad_lines=False,
                                    encoding="latin", header=None)
        item_features.rename(columns={0: self.ITEM_ID_COL, 1: 'title', 2: 'date'}, inplace=True)
        genres = pd.read_csv(self._genres_file, sep='|', header=None)
        genre_names = list(genres[0].values)
        item_features.rename(columns={i: genre_names[i - 5] for i in range(5, 24)}, inplace=True)
        item_features['year'] = pd.to_datetime(item_features['date']).apply(lambda x: x.year)
        return item_features

    def _get_user_features(self):
        users = pd.read_csv(self._user_file, sep='|', header=None)
        users.columns = [self.USER_ID_COL, 'age', 'gender', 'profession', 'zip']
        return users
