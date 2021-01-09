import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from kerastuner import HyperModel, RandomSearch
from tensorflow import keras
from tensorflow.keras.callbacks import Callback

from bias_tree import get_metric_bias_tree_for_model


class RankingModel(tf.keras.Model):

    def __init__(self, unique_user_ids: np.array, unique_item_ids: np.array, embedding_size: int,
                 regularization_coef: float = 1e-6, dropout_rate: float = 0.0):
        super().__init__()
        self.embedding_size = embedding_size
        self.user_embedding = tf.keras.layers.Embedding(len(unique_user_ids), embedding_size,
                                                        embeddings_initializer="he_normal",
                                                        embeddings_regularizer=keras.regularizers.l2(
                                                            regularization_coef)
                                                        )
        self.item_embedding = tf.keras.layers.Embedding(len(unique_item_ids), embedding_size,
                                                        embeddings_initializer="he_normal",
                                                        embeddings_regularizer=keras.regularizers.l2(
                                                            regularization_coef)
                                                        )
        self.ratings = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu",
                                  kernel_regularizer=keras.regularizers.l2(
                                      regularization_coef),
                                  activity_regularizer=keras.regularizers.l2(
                                      regularization_coef),
                                  bias_regularizer=keras.regularizers.l2(
                                      regularization_coef), ),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=keras.regularizers.l2(
                regularization_coef),
                                  activity_regularizer=keras.regularizers.l2(
                                      regularization_coef),
                                  bias_regularizer=keras.regularizers.l2(
                                      regularization_coef)),
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(1)
        ])

    def call(self, inputs):
        user_embedding = self.user_embedding(inputs[:, 0])
        item_embedding = self.item_embedding(inputs[:, 1])

        return self.ratings(tf.concat([user_embedding, item_embedding], axis=1))


class BiasEvaluationCallback(Callback):
    def __init__(self, train_data, validation_data, data, metric='squared_error', min_child_node_size=1000,
                 max_depth=3, interval=5, rating_col='rating', user_col='user_id',
                 item_col='item_id'):
        super(Callback, self).__init__()
        self.interval = interval
        self.X_train = train_data
        self.X_val = validation_data
        self.data = data
        self.metric = metric
        self.min_child_node_size = min_child_node_size
        self.max_depth = max_depth
        self.bias_results = []
        self.rating_col = rating_col
        self.user_col = user_col
        self.item_col = item_col

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            self._get_bias_tree(self.X_train, 'train', epoch)
            self._get_bias_tree(self.X_val, 'val', epoch)
            self.bias_results.append({'epoch': epoch,
                                      'value': logs['loss'],
                                      'metric': 'avg-train',
                                      })
            self.bias_results.append({'epoch': epoch,
                                      'value': logs['val_loss'],
                                      'metric': 'avg-val',
                                      })

    def _get_bias_tree(self, X, data_name, epoch):
        bias_tree = get_metric_bias_tree_for_model(self.model, X, self.data.attributes_dict,
                                                   metric_name=self.metric,
                                                   min_child_node_size=self.min_child_node_size,
                                                   max_depth=self.max_depth, rating_col=self.rating_col,
                                                   user_col=self.user_col, item_col=self.item_col)
        print("interval evaluation - min node {}, max node {}".format(bias_tree.min_metric_value,
                                                                      bias_tree.max_metric_value))
        self.bias_results.append({'epoch': epoch,
                                  'value': bias_tree.min_metric_value,
                                  'metric': '{}-min node'.format(data_name)
                                  })
        self.bias_results.append({'epoch': epoch,
                                  'value': bias_tree.max_metric_value,
                                  'metric': '{}-max node'.format(data_name),
                                  })
        plt.show()


class RankingRecommenderHyperParamSearch(HyperModel):
    def __init__(self, user_ids: list, item_ids: list):
        super().__init__()
        self.unique_user_ids = np.asarray(user_ids)
        self.unique_item_ids = np.asarray(item_ids)

    def build(self, hyperparam_tuner):
        model = RankingModel(self.unique_user_ids, self.unique_item_ids,
                             embedding_size=hyperparam_tuner.Int('embedding_size',
                                                                 min_value=128,
                                                                 max_value=128,
                                                                 step=8),

                             regularization_coef=hyperparam_tuner.Choice('regularization_coef',
                                                                         values=[1e-5]),
                             )

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(lr=hyperparam_tuner.Choice('learning_rate',
                                                                       values=[1e-3 ])
                                            ))
        return model


class EpochRandomSearch(RandomSearch):
    def run_trial(self, trial, *args, **kwargs):
        kwargs['epochs'] = trial.hyperparameters.Int('epochs', min_value=10, max_value=50, step=10)
        super().run_trial(trial, *args, **kwargs)


def fit_recommendation_model(train_data: pd.DataFrame, val_data: pd.DataFrame, user_ids: list, item_ids: list,
                             batch_size: int = 64, epochs: int = 5, embedding_size: int = 20, callbacks=[],
                             lr: float = 0.001, regularization_coef: float = 1e-6, metrics=[],
                             rating_col='rating', user_col='user_id',
                             item_col='item_id') -> (RankingModel,
                                                     dict):
    model = RankingModel(user_ids, item_ids, embedding_size, regularization_coef=regularization_coef)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(lr=lr),
        metrics=metrics
    )
    model, history = retrain_recommendation_model(train_data=train_data, val_data=val_data, model=model,
                                                  batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                                                  rating_col=rating_col, user_col=user_col, item_col=item_col)

    return model, history


def retrain_recommendation_model(train_data: pd.DataFrame, val_data: pd.DataFrame,
                                 model: RankingModel, retrain_embeddings: bool = False,
                                 batch_size: int = 64, epochs: int = 5,
                                 plot_history: bool = True, callbacks=[], rating_col='rating', user_col='user_id',
                                 item_col='item_id') -> (RankingModel, dict):
    model.user_embedding.trainable = retrain_embeddings
    model.item_embedding.trainable = retrain_embeddings
    # callbacks = []
    # tf.keras.callbacks.EarlyStopping(patience=2),]
    history = model.fit(
        x=train_data[[user_col, item_col]].values,
        y=train_data[rating_col].values,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_data[[user_col, item_col]].values,
                         val_data[rating_col].values),
        callbacks=callbacks
    )
    if plot_history:
        plt.plot(history.history["loss"])
        plt.plot(history.history["val_loss"])
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epoch")
        plt.legend(["train", "validation"], loc="upper left")
        plt.show()

    return model, history


def tune_recommendation_hyperparams(train_data: pd.DataFrame, val_data: pd.DataFrame, user_ids: list, item_ids: list,
                                    batch_size: int = 64, epochs: int = 5, max_trials: int = 10,
                                    project_suffix='', logdir: str = 'logs', rating_col='rating', user_col='user_id',
                                    item_col='item_id') -> RankingRecommenderHyperParamSearch:
    model = RankingRecommenderHyperParamSearch(user_ids, item_ids)
    tuner = EpochRandomSearch(
        model,
        objective='val_loss',
        max_trials=max_trials,
        directory=logdir,
        project_name=f'recommeder-debias-{project_suffix}')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),
    ]
    tuner.search(
        x=train_data[[user_col, item_col]].values,
        y=train_data[rating_col].values,
        batch_size=batch_size,
        # epochs=epochs,
        verbose=1,
        validation_data=(val_data[[user_col, item_col]].values,
                         val_data[rating_col].values),
        callbacks=callbacks
    )
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    # print(best_hps.values['epochs'])
    model = tuner.hypermodel.build(best_hps)
    model.fit(
        x=train_data[[user_col, item_col]].values,
        y=train_data[rating_col].values,
        batch_size=batch_size,
        epochs=best_hps.values['epochs'],
        validation_data=(val_data[[user_col, item_col]].values,
                         val_data[rating_col].values),
        callbacks=callbacks
    )
    return model
