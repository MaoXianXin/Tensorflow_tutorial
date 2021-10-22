import os

import tensorflow as tf
import tensorflow_hub as hub

import gdown
import numpy as np
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# gdown.download(
#     url='https://drive.google.com/uc?id=1Ag0jd21oRwJhVFIBohmX_ogeojVtapLy',
#     output='bard.zip',
#     quiet=False
# )

module_path = "text_module"
embedding_layer = hub.KerasLayer(module_path, trainable=False)

embedding_layer(['বাস', 'বসবাস', 'ট্রেন', 'যাত্রী', 'ট্রাক'])

dir_names = ['economy', 'sports', 'entertainment', 'state', 'international']

file_paths = []
labels = []
for i, dir in enumerate(dir_names):
    file_names = ["/".join([dir, name]) for name in os.listdir(dir)]
    file_paths += file_names
    labels += [i] * len(os.listdir(dir))

np.random.seed(42)
permutation = np.random.permutation(len(file_paths))

file_paths = np.array(file_paths)[permutation]
labels = np.array(labels)[permutation]

train_frac = 0.8
train_size = int(len(file_paths) * train_frac)

# plot training vs validation distribution
plt.subplot(1, 2, 1)
plt.hist(labels[0:train_size])
plt.title("Train labels")
plt.subplot(1, 2, 2)
plt.hist(labels[train_size:])
plt.title("Validation labels")
plt.tight_layout()
plt.show()


def load_file(path, label):
    return tf.io.read_file(path), label


def make_datasets(train_size):
    batch_size = 256

    train_files = file_paths[:train_size]
    train_labels = labels[:train_size]
    train_ds = tf.data.Dataset.from_tensor_slices((train_files, train_labels))
    train_ds = train_ds.map(load_file).shuffle(5000)
    train_ds = train_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    test_files = file_paths[train_size:]
    test_labels = labels[train_size:]
    test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_labels))
    test_ds = test_ds.map(load_file)
    test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    return train_ds, test_ds


train_data, validation_data = make_datasets(train_size)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=[], dtype=tf.string),
        embedding_layer,
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(5),
    ])
    model.compile(loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                  optimizer="adam", metrics=['accuracy'])
    return model


model = create_model()
# Create earlystopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3)

history = model.fit(train_data,
                    validation_data=validation_data,
                    epochs=5,
                    callbacks=[early_stopping_callback])

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_pred = model.predict(validation_data)
y_pred = np.argmax(y_pred, axis=1)

samples = file_paths[0:3]
for i, sample in enumerate(samples):
    f = open(sample)
    text = f.read()
    print(text[0:100])
    print("True Class: ", sample.split("/")[0])
    print("Predicted Class: ", dir_names[y_pred[i]])
    f.close()

y_true = np.array(labels[train_size:])
print(classification_report(y_true, y_pred, target_names=dir_names))
