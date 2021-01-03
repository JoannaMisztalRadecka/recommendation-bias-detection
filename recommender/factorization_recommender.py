import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from kerastuner import HyperModel, RandomSearch
from tensorflow import keras

class FactorizationRecommender(keras.Model):
    def __init__(self, num_users: int, num_items: int, embedding_size: int, regularization_coef: float = 1e-6,
                 **kwargs):
        super().__init__(**kwargs)
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_size = embedding_size
        self.user_embedding = keras.layers.Embedding(
            num_users,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(regularization_coef),
        )
        self.user_bias = keras.layers.Embedding(num_users, 1)
        self.item_embedding = keras.layers.Embedding(
            num_items,
            embedding_size,
            embeddings_initializer="he_normal",
            embeddings_regularizer=keras.regularizers.l2(regularization_coef),
        )
        self.item_bias = keras.layers.Embedding(num_items, 1)

    def call(self, inputs):
        user_vector = self.user_embedding(inputs[:, 0])
        user_bias = self.user_bias(inputs[:, 0])
        item_vector = self.item_embedding(inputs[:, 1])
        item_bias = self.item_bias(inputs[:, 1])
        dot_user_item = tf.tensordot(user_vector, item_vector, 2)
        x = dot_user_item + user_bias + item_bias
        return x  # tf.nn.sigmoid(x)


class FactorizationRecommenderHyperParamSearch(HyperModel):
    def __init__(self, num_users: int, num_items: int):
        super().__init__()
        self.num_users = num_users
        self.num_items = num_items

    def build(self, hyperparam_tuner):
        model = FactorizationRecommender(self.num_users, self.num_items,
                                         embedding_size=hyperparam_tuner.Int('embedding_size',
                                                                             min_value=16,
                                                                             max_value=64,
                                                                             step=16),
                                         regularization_coef=hyperparam_tuner.Choice('regularization_coef',
                                                                                     values=[1e-2, 1e-4, 1e-6]))

        model.compile(
            loss=tf.keras.losses.MeanSquaredError(),
            optimizer=keras.optimizers.Adam(lr=hyperparam_tuner.Choice('learning_rate',
                                                                       values=[1e-2, 1e-3, 1e-4])
                                            ))
        return model


def fit_recommendation_model(train_data: pd.DataFrame, val_data: pd.DataFrame, num_users: int, num_items: int,
                             batch_size: int = 64, epochs: int = 5, embedding_size: int = 20,
                             lr: float = 0.001, regularization_coef: float = 1e-6) -> FactorizationRecommender:
    model = FactorizationRecommender(num_users, num_items, embedding_size, regularization_coef=regularization_coef)
    model.compile(
        loss=tf.keras.losses.MeanSquaredError(),
        optimizer=keras.optimizers.Adam(lr=lr)
    )
    retrain_recommendation_model(train_data=train_data, val_data=val_data, model=model,
                                 batch_size=batch_size, epochs=epochs)

    return model


def retrain_recommendation_model(train_data: pd.DataFrame, val_data: pd.DataFrame,
                                 model: FactorizationRecommender, retrain_embeddings: bool = False,
                                 batch_size: int = 64, epochs: int = 5, plot_history:bool=True) -> FactorizationRecommender:
    model.user_embedding.trainable = retrain_embeddings
    model.item_embedding.trainable = retrain_embeddings
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2),]
    history = model.fit(
        x=train_data[["user_id", "item_id"]].values,
        y=train_data["rating"].values,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(val_data[["user_id", "item_id"]].values,
                         val_data["rating"].values),
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

    return model


def tune_recommendation_hyperparams(train_data: pd.DataFrame, val_data: pd.DataFrame,num_users: int, num_items: int,
                                    batch_size: int = 64, epochs: int = 5) -> FactorizationRecommenderHyperParamSearch:
    model = FactorizationRecommenderHyperParamSearch(num_users, num_items)
    tuner = RandomSearch(
        model,
        objective='val_loss',
        max_trials=20,
        directory='hyperparams',
        project_name='recommeder-debias')
    callbacks = [
        tf.keras.callbacks.EarlyStopping(patience=2), ]
    history = tuner.search(
        x=train_data[["user_id", "item_id"]].values,
        y=train_data["rating"].values,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(val_data[["user_id", "item_id"]].values,
                         val_data["rating"].values),
        callbacks=callbacks
    )

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

    return model
