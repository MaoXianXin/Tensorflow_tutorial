from typing import Dict, Text

import numpy as np
import tensorflow_datasets as tfds

import tensorflow_recommenders as tfrs

import re
import string
import tensorflow as tf

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

# Ratings data.
ratings = tfds.load("movielens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movielens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])

tf.random.set_seed(42)
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

movie_titles = movies.batch(1_000)
user_ids = ratings.batch(1_000_000).map(lambda x: x["user_id"])

unique_movie_titles = np.unique(np.concatenate(list(movie_titles)))
unique_user_ids = np.unique(np.concatenate(list(user_ids)))

embedding_dimension = 32

max_features = 3000
sequence_length = 32
user_sequence_length = 32


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
    return tf.strings.regex_replace(stripped_html,
                                    '[%s]' % re.escape(string.punctuation),
                                    '')


vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

user_vectorize_layer = TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=user_sequence_length)

vectorize_layer.adapt(unique_movie_titles)
user_vectorize_layer.adapt(unique_user_ids)

user_model = tf.keras.Sequential([
    user_vectorize_layer,
    # We add an additional embedding to account for unknown tokens.
    tf.keras.layers.Embedding(len(user_vectorize_layer.get_vocabulary()) + 1, embedding_dimension),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalMaxPooling1D(),
])

movie_model = tf.keras.Sequential([
    vectorize_layer,
    tf.keras.layers.Embedding(len(vectorize_layer.get_vocabulary()) + 1, embedding_dimension),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.GlobalMaxPooling1D(),
])


metrics = tfrs.metrics.FactorizedTopK(
    candidates=movies.batch(128).map(movie_model)
)

task = tfrs.tasks.Retrieval(
    metrics=metrics
)


class MovielensModel(tfrs.Model):

    def __init__(self, user_model, movie_model):
        super().__init__()
        self.movie_model: tf.keras.Model = movie_model
        self.user_model: tf.keras.Model = user_model
        self.task: tf.keras.layers.Layer = task

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        # We pick out the user features and pass them into the user model.
        user_embeddings = self.user_model(features["user_id"])
        # And pick out the movie features and pass them into the movie model,
        # getting embeddings back.
        positive_movie_embeddings = self.movie_model(features["movie_title"])

        # The task computes the loss and the metrics.
        return self.task(user_embeddings, positive_movie_embeddings)


model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(1024).cache()

model.fit(cached_train, epochs=100)

model.evaluate(cached_test, return_dict=True)
