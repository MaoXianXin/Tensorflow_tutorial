import numpy as np
from tabulate import tabulate
from matplotlib import pyplot as plt

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam

import tensorflow_similarity as tfsim  # main package

from tensorflow_similarity.utils import tf_cap_memory
from tensorflow_similarity.layers import MetricEmbedding  # row wise L2 norm
from tensorflow_similarity.losses import MultiSimilarityLoss  # specialized similarity loss
from tensorflow_similarity.models import SimilarityModel  # TF model with additional features
from tensorflow_similarity.samplers import MultiShotMemorySampler  # sample data
from tensorflow_similarity.samplers import select_examples  # select n example per class
from tensorflow_similarity.visualization import viz_neigbors_imgs  # neigboors vizualisation
from tensorflow_similarity.visualization import confusion_matrix  # matching performance

tfsim.utils.tf_cap_memory()

print('TensorFlow:', tf.__version__)
print('TensorFlow Similarity', tfsim.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

CLASSES = [2, 3, 1, 7, 9, 6, 8, 5, 0, 4]
NUM_CLASSES = 6  # @param {type: "slider", min: 1, max: 10}
CLASSES_PER_BATCH = NUM_CLASSES
EXAMPLES_PER_CLASS = 6  # @param {type:"integer"}
STEPS_PER_EPOCH = 1000  # @param {type:"integer"}

sampler = MultiShotMemorySampler(x_train, y_train,
                                 classes_per_batch=CLASSES_PER_BATCH,
                                 examples_per_class_per_batch=EXAMPLES_PER_CLASS,
                                 class_list=CLASSES[:NUM_CLASSES],  # Only use the first 6 classes for training.
                                 steps_per_epoch=STEPS_PER_EPOCH)


def get_model():
    inputs = layers.Input(shape=(28, 28, 1))
    x = layers.experimental.preprocessing.Rescaling(1 / 255)(inputs)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.Conv2D(32, 3, activation='relu')(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Conv2D(64, 3, activation='relu')(x)
    x = layers.Flatten()(x)
    # smaller embeddings will have faster lookup times while a larger embedding will improve the accuracy up to a point.
    outputs = MetricEmbedding(64)(x)
    return SimilarityModel(inputs, outputs)


model = get_model()
print(model.summary())

distance = 'cosine'  # @param ["cosine", "L2", "L1"]{allow-input: false}
loss = MultiSimilarityLoss(distance=distance)

LR = 0.0001  # @param {type:"number"}
# model = get_model()
model.compile(optimizer=Adam(LR), loss=loss)

EPOCHS = 10  # @param {type:"integer"}
history = model.fit(sampler, epochs=EPOCHS, validation_data=(x_test, y_test))

# expect loss: 0.014 / val_loss: 1.3
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['loss', 'val_loss'])
plt.title(f'Loss: {loss.name} - LR: {LR}')
plt.show()

x_index, y_index = select_examples(x_train, y_train, CLASSES, 20)
model.reset_index()
model.index(x_index, y_index, data=x_index)

# re-run to test on other examples
num_neighboors = 5

# select
x_display, y_display = select_examples(x_test, y_test, CLASSES, 1)

# lookup nearest neighbors in the index
nns = model.lookup(x_display, k=num_neighboors)

# display
for idx in np.argsort(y_display):
    viz_neigbors_imgs(x_display[idx], y_display[idx], nns[idx],
                      fig_size=(16, 2), cmap='Greys')

num_calibration_samples = 1000  # @param {type:"integer"}
calibration = model.calibrate(
    x_train[:num_calibration_samples],
    y_train[:num_calibration_samples],
    extra_metrics=['precision', 'recall', 'binary_accuracy'],
    verbose=1
)

fig, ax = plt.subplots()
x = calibration.thresholds['distance']
ax.plot(x, calibration.thresholds['precision'], label='precision')
ax.plot(x, calibration.thresholds['recall'], label='recall')
ax.plot(x, calibration.thresholds['f1'], label='f1 score')
ax.legend()
ax.set_title("Metric evolution as distance increase")
ax.set_xlabel('Distance')
plt.show()

fig, ax = plt.subplots()
ax.plot(calibration.thresholds['recall'], calibration.thresholds['precision'])
ax.set_title("Precision recall curve")
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
plt.show()

num_matches = 10  # @param {type:"integer"}

matches = model.match(x_test[:num_matches], cutpoint='optimal')
rows = []
for idx, match in enumerate(matches):
    rows.append([match, y_test[idx], match == y_test[idx]])
print(tabulate(rows, headers=['Predicted', 'Expected', 'Correct']))

# used to label in images in the viz_neighbors_imgs plots
# note we added a 11th classes for unknown
labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Unknown"]
num_examples_per_class = 1000
cutpoint = 'optimal'

x_confusion, y_confusion = select_examples(x_test, y_test, CLASSES, num_examples_per_class)

matches = model.match(x_confusion, cutpoint=cutpoint, no_match_label=10)
confusion_matrix(matches, y_confusion, labels=labels, title='Confusin matrix for cutpoint:%s' % cutpoint)

print(model.index_summary())

# save the model and the index
save_path = 'models/hello_world'  # @param {type:"string"}
model.save(save_path)

# reload the model
reloaded_model = load_model(save_path)
# reload the index
reloaded_model.load_index(save_path)

# check the index is back
print(reloaded_model.index_summary())

# re-run to test on other examples
num_neighboors = 5

# select
x_display, y_display = select_examples(x_test, y_test, CLASSES, 1)

# lookup the nearest neighbors
nns = model.lookup(x_display, k=num_neighboors)

# display
for idx in np.argsort(y_display):
    viz_neigbors_imgs(x_display[idx], y_display[idx], nns[idx],
                      fig_size=(16, 2), cmap='Greys')
