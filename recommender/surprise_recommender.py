import pandas as pd
from sklearn.model_selection import ParameterGrid
import surprise
from surprise.prediction_algorithms import AlgoBase
from surprise.trainset import Trainset


class SurpriseRecommender:
    def __init__(self, model: AlgoBase):
        self.model = model

    def fit(self, trainset: Trainset):
        self.model.fit(trainset)

    def predict(self, testset: pd.DataFrame):
        return testset.apply(lambda x: self.model.predict(x['user_id'], x['item_id']).est, 1)


def random_search_fit_surprise_recommendation_model(ratings: pd.DataFrame, surprise_model_cls, params_grid: dict,
                                                    n_iter: int = 50, metric: str='mae') -> SurpriseRecommender:
    param_grid_len = len(list(ParameterGrid(params_grid)))
    n_iter = min(n_iter, param_grid_len)
    dataset = build_surprise_trainset(ratings, min_rating=min(ratings['rating']), max_rating=max(ratings['rating']))
    search = surprise.model_selection.RandomizedSearchCV(surprise_model_cls, param_distributions=params_grid,
                                                         n_iter=n_iter, n_jobs=-1, refit=True)
    search.fit(dataset)
    print(search.best_params[metric])
    recommender = SurpriseRecommender(model=search)

    return recommender


def build_surprise_trainset(ratings: pd.DataFrame, min_rating: int = 1, max_rating: int = 5):
    reader = surprise.dataset.Reader(rating_scale=(min_rating, max_rating))
    dataset = surprise.dataset.Dataset.load_from_df(ratings[['user_id', 'item_id', 'rating']],
                                                    reader=reader)
    return dataset
