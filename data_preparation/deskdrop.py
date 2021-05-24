import os
from urllib.parse import urlparse

import pandas as pd
import pycountry
from user_agents import parse

from data_preparation.recommender_dataset import RecommenderDataset


class DeskdropData(RecommenderDataset):
    """
    Prepares Deskdrop dataset for recommendation model training.
    """

    NAME = 'Deskdrop'
    USER_ID_COL = 'personId'
    ITEM_ID_COL = 'contentId'
    RATING_COL = 'rating'
    MIN_RATING = 1
    MAX_RATING = 5

    def __init__(self, data_path: str, min_user_interactions: int = 10, min_item_interactions: int = 10,
                 sample: int = None, n_bins: int = 5,bucket_labels: list=None):
        """
        :param data_path: Path with the dataset.
        """
        super().__init__(data_path, sample=sample, min_user_interactions=min_user_interactions,
                         min_item_interactions=min_item_interactions, n_bins=n_bins, bucket_labels=bucket_labels)
        self._ratings_file = os.path.join(data_path, 'users_interactions.csv.zip')
        self._item_file = os.path.join(data_path, 'shared_articles.csv.zip')

    def _get_ratings(self):
        ratings = pd.read_csv(self._ratings_file).fillna('')
        event_rating = {'VIEW': 1,
                        'LIKE': 2,
                        'COMMENT CREATED': 3,
                        'FOLLOW': 4,
                        'BOOKMARK': 5}
        ratings[self.RATING_COL] = ratings['eventType'].apply(lambda x: event_rating.get(x, 1))
        ratings['ua'] = ratings['userAgent'].apply(parse)
        ratings['device'] = ratings['ua'].apply(lambda x: "Device:" + x.device.family)
        ratings['os'] = ratings['ua'].apply(lambda x: "OS:" + x.os.family)
        ratings['country'] = ratings['userCountry'].apply(self._get_country_name)
        return ratings

    @staticmethod
    def _get_country_name(abbr):
        country = pycountry.countries.get(alpha_2=abbr)

        if country:
            return 'Country:' + country.name
        return ''

    def _get_item_features(self):
        item_features = pd.read_csv(self._item_file)
        item_features['host'] = 'URL:' + item_features['url'].apply(lambda x: urlparse(x).hostname)

        return item_features.drop_duplicates()

    @property
    def attributes_categorical(self):
        return ['lang', 'country', 'device', 'os', 'host']

    @property
    def attributes_continuous(self):
        return ['user activity', 'item popularity', ]

    def _get_user_features(self):
        return None
