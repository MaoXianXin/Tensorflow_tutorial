import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import zipfile

from sklearn import model_selection

import os
import pathlib

SENTIMENT_LABELS = [
    "negative", "somewhat negative", "neutral", "somewhat positive", "positive"
]


# Add a column with readable values representing the sentiment.
def add_readable_labels_column(df, sentiment_value_column):
    df["SentimentLabel"] = df[sentiment_value_column].replace(
        range(5), SENTIMENT_LABELS)


# Download data from Kaggle and create a DataFrame.
def load_data_from_zip(path):
    with zipfile.ZipFile(path, "r") as zip_ref:
        name = zip_ref.namelist()[0]
        with zip_ref.open(name) as zf:
            return pd.read_csv(zf, sep="\t", index_col=0)


# The data does not come with a validation set so we'll create one from the
# training set.
def get_data(competition, train_file, test_file, validation_set_ratio=0.1):
    competition_path = '/home/csdn/sentiment-analysis-on-movie-reviews/'

    train_df = load_data_from_zip(competition_path + train_file)
    test_df = load_data_from_zip(competition_path + test_file)

    # Add a human readable label.
    add_readable_labels_column(train_df, "Sentiment")

    # We split by sentence ids, because we don't want to have phrases belonging
    # to the same sentence in both training and validation set.
    train_indices, validation_indices = model_selection.train_test_split(
        np.unique(train_df["SentenceId"]),
        test_size=validation_set_ratio,
        random_state=0)

    validation_df = train_df[train_df["SentenceId"].isin(validation_indices)]
    train_df = train_df[train_df["SentenceId"].isin(train_indices)]
    print("Split the training data into %d training and %d validation examples." %
          (len(train_df), len(validation_df)))

    return train_df, validation_df, test_df


train_df, validation_df, test_df = get_data(
    "sentiment-analysis-on-movie-reviews",
    "train.tsv.zip", "test.tsv.zip")


class MyModel(tf.keras.Model):
    def __init__(self, hub_url):
        super().__init__()
        self.hub_url = hub_url
        self.embed = hub.load(self.hub_url).signatures['default']
        self.sequential = tf.keras.Sequential([
            tf.keras.layers.Dense(500),
            tf.keras.layers.Dense(100),
            tf.keras.layers.Dense(5),
        ])

    def call(self, inputs):
        phrases = inputs['Phrase'][:, 0]
        embedding = 5 * self.embed(phrases)['default']
        return self.sequential(embedding)

    def get_config(self):
        return {"hub_url": self.hub_url}


model = MyModel("https://tfhub.dev/google/nnlm-en-dim128/1")
model.compile(
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=tf.optimizers.Adam(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")])

history = model.fit(x=dict(train_df), y=train_df['Sentiment'],
                    validation_data=(dict(validation_df), validation_df['Sentiment']),
                    epochs=25)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()

train_eval_result = model.evaluate(dict(train_df), train_df['Sentiment'])
validation_eval_result = model.evaluate(dict(validation_df), validation_df['Sentiment'])

print(f"Training set accuracy: {train_eval_result[1]}")
print(f"Validation set accuracy: {validation_eval_result[1]}")

predictions = model.predict(dict(validation_df))
predictions = tf.argmax(predictions, axis=-1)

cm = tf.math.confusion_matrix(validation_df['Sentiment'], predictions)
cm = cm / cm.numpy().sum(axis=1)[:, tf.newaxis]
sns.heatmap(
    cm, annot=True,
    xticklabels=SENTIMENT_LABELS,
    yticklabels=SENTIMENT_LABELS)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
