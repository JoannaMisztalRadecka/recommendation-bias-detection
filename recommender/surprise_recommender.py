import pandas as pd
import surprise
from sklearn.model_selection import ParameterGrid
from surprise.prediction_algorithms import AlgoBase
from surprise.trainset import Trainset
from surprise.dataset import DatasetAutoFolds

class SurpriseRecommender:
    def __init__(self, model: AlgoBase, user_col='user_id',
                 item_col='item_id'):
        self.model = model
        self._user_col = user_col
        self._item_col = item_col

    def fit(self, trainset: Trainset):
        self.model.fit(trainset)

    def predict(self, testset: pd.DataFrame):
        return testset.apply(lambda x: self.model.predict(x[self._user_col], x[self._item_col]).est, 1)


def random_search_fit_surprise_recommendation_model(ratings: pd.DataFrame, surprise_model_cls, params_grid: dict,
                                                    n_iter: int = 50, metric: str = 'mae', rating_col='rating',
                                                    user_col='user_id',
                                                    item_col='item_id') -> SurpriseRecommender:
    param_grid_len = len(list(ParameterGrid(params_grid)))
    n_iter = min(n_iter, param_grid_len)
    dataset = build_surprise_trainset(ratings, min_rating=min(ratings[rating_col]), max_rating=max(ratings[rating_col]),
                                      rating_col=rating_col, user_col=user_col, item_col=item_col)
    search = surprise.model_selection.RandomizedSearchCV(surprise_model_cls, param_distributions=params_grid,
                                                         n_iter=n_iter, n_jobs=-1, refit=True)
    search.fit(dataset)
    print(search.best_params[metric])
    recommender = SurpriseRecommender(model=search)

    return recommender

def fit_surprise_recommendation_model(ratings: pd.DataFrame, surprise_model_cls, params: dict,
                                                    n_iter: int = 50, metric: str = 'mae', rating_col='rating',
                                                    user_col='user_id',
                                                    item_col='item_id') -> SurpriseRecommender:
    dataset = build_surprise_trainset(ratings, min_rating=min(ratings[rating_col]), max_rating=max(ratings[rating_col]),
                                      rating_col=rating_col, user_col=user_col, item_col=item_col)
    trainset = DatasetAutoFolds.build_full_trainset(dataset)
    model = surprise_model_cls(**params)
    model.fit(trainset)
    recommender = SurpriseRecommender(model=model)

    return recommender


def build_surprise_trainset(ratings: pd.DataFrame, min_rating: int = 1, max_rating: int = 5,
                            rating_col='rating', user_col='user_id', item_col='item_id'):
    reader = surprise.dataset.Reader(rating_scale=(min_rating, max_rating))
    dataset = surprise.dataset.Dataset.load_from_df(ratings[[user_col, item_col, rating_col]],
                                                    reader=reader)
    return dataset
