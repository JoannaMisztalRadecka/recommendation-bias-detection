import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras


class MFRecommender(keras.Model):
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
        return tf.nn.sigmoid(x)


def fit_recommendation_model(train_data: pd.DataFrame, val_data: pd.DataFrame, num_users: int, num_items: int,
                             batch_size: int = 64, epochs: int = 5, embedding_size: int = 20,
                             lr: float = 0.001) -> MFRecommender:
    model = MFRecommender(num_users, num_items, embedding_size)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(), optimizer=keras.optimizers.Adam(lr=lr)
    )
    retrain_recommendation_model(train_data=train_data, val_data=val_data, model=model,
                                 batch_size=batch_size, epochs=epochs)

    return model


def retrain_recommendation_model(train_data: pd.DataFrame, val_data: pd.DataFrame,
                                 model: MFRecommender, retrain_embeddings: bool = False,
                                 batch_size: int = 64, epochs: int = 5) -> MFRecommender:
    model.user_embedding.trainable = retrain_embeddings
    model.item_embedding.trainable = retrain_embeddings
    history = model.fit(
        x=train_data[["user_id", "item_id"]].values,
        y=train_data["rating_scaled"].values,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(val_data[["user_id", "item_id"]].values,
                         val_data["rating_scaled"].values),
    )

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epoch")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

    return model
