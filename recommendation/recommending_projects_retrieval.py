from typing import Dict, Text
import numpy as np
import tensorflow as tf

import tensorflow_recommenders as tfrs
import pandas as pd

projects = pd.read_csv('../test.csv', usecols=['user_id', 'project_name'])
print(len(projects))

# projects = projects.head(100000)

train_num = int(0.8 * len(projects))

unique_movie_titles = np.asarray(projects['project_name'].drop_duplicates())
print('movie_title num: ', len(unique_movie_titles))
unique_user_ids = np.asarray(projects['user_id'].drop_duplicates())
print('user_num: ', len(unique_user_ids))

ratings = projects
movies = unique_movie_titles

ratings = tf.data.Dataset.from_tensor_slices(ratings)
movies = tf.data.Dataset.from_tensor_slices(movies)

tf.random.set_seed(42)
shuffled = ratings.shuffle(train_num, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(train_num)
test = shuffled.skip(train_num).take(len(projects) - train_num)

embedding_dimension = 32

user_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_user_ids, mask_token=None
    ),
    tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
])


movie_model = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.StringLookup(
        vocabulary=unique_movie_titles, mask_token=None
    ),
    tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
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
        user_embeddings = self.user_model(features[:, 0])
        positive_movie_embeddings = self.movie_model(features[:, 1])

        return self.task(user_embeddings, positive_movie_embeddings)


model = MovielensModel(user_model, movie_model)
model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

cached_train = train.shuffle(100_000).batch(8192).cache()
cached_test = test.batch(4096).cache()


model.fit(cached_train, epochs=3)

model.evaluate(cached_test, return_dict=True)