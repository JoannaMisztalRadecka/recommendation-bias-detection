import os

import pandas as pd

from data_preparation.recommender_dataset import RecommenderDataset


class BookCrossingData(RecommenderDataset):
    NAME = 'BookCrossing'
    USER_ID_COL = 'User-ID'
    ITEM_ID_COL = 'ISBN'
    RATING_COL = 'Book-Rating'
    SENSITIVE_ATTRIBUTES = ['country', 'age_group']
    SPARSE_FEATURES = [USER_ID_COL, ITEM_ID_COL, "age_group", "country", "Book-Author", 'year',
                       "Publisher"]
    DATA_PATH = 'data/book-crossing/'
    MIN_RATING = 0
    MAX_RATING = 10

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5, bucket_labels: list=None):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins, bucket_labels=bucket_labels)
        self._ratings_file = os.path.join(data_path, 'BX-Book-Ratings.csv')
        self._item_file = os.path.join(data_path, 'BX-Books.csv')
        self._user_file = os.path.join(data_path, 'BX-Users.csv')

    def _get_ratings(self):
        data = pd.read_csv(self._ratings_file, sep=';', error_bad_lines=False, warn_bad_lines=False,
                           encoding="latin").fillna(0)

        return data[data[self.RATING_COL] > 0]

    def _get_user_features(self):
        user_features = pd.read_csv(self._user_file, sep=';', error_bad_lines=False, warn_bad_lines=False,
                                    encoding="latin")
        user_features['Age'].fillna(-1, inplace=True)
        # user_features['age_group'] = user_features['Age'].apply(
        #     lambda x: '' if str(x) == 'nan' or x < 10 or x > 90 else 'Age:{}0s'.format(round(x / 10)))
        user_features['country'] = user_features['Location'].apply(lambda x: 'Country:' + x.split(', ')[-1])
        user_countries = user_features.country.value_counts()
        user_features = user_features.merge(user_countries[user_countries >= 1000].to_frame(), left_on='country',
                                            right_index=True)
        return user_features.set_index(self.USER_ID_COL).fillna('unknown')

    def _get_item_features(self):
        item_features = pd.read_csv(self._item_file, sep=';', error_bad_lines=False, warn_bad_lines=False,
                                    encoding="latin", )
        item_features['Year-Of-Publication'] = pd.to_numeric(item_features['Year-Of-Publication'],
                                                             errors='coerce').fillna(-1)
        # item_features['year'] = item_features['Year-Of-Publication'].apply(
        #     lambda x: '' if x == 0 else 'Year:{}0s'.format(str(x)[:-1]))
        return item_features.set_index(self.ITEM_ID_COL)

    @property
    def attributes_categorical(self):
        return ["country", ]  # 'Publisher']

    @property
    def attributes_continuous(self):
        return ['user activity', 'item popularity', 'Age', 'Year-Of-Publication',]
