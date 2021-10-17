import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

dataset, info = tfds.load('cassava', with_info=True)

# Extend the cassava dataset classes with 'unknown'
class_names = info.features['label'].names + ['unknown']

# Map the class names to human readable names
name_map = dict(
    cmd='Mosaic Disease',
    cbb='Bacterial Blight',
    cgm='Green Mite',
    cbsd='Brown Streak Disease',
    healthy='Healthy',
    unknown='Unknown')

print(len(class_names), 'classes:')
print(class_names)
print([name_map[name] for name in class_names])


def plot(examples, predictions=None):
    # Get the images, labels, and optionally predictions
    images = examples['image']
    labels = examples['label']
    batch_size = len(images)
    if predictions is None:
        predictions = batch_size * [None]

    # Configure the layout of the grid
    x = np.ceil(np.sqrt(batch_size))
    y = np.ceil(batch_size / x)
    fig = plt.figure(figsize=(x * 6, y * 7))

    for i, (image, label, prediction) in enumerate(zip(images, labels, predictions)):
        # Render the image
        ax = fig.add_subplot(x, y, i + 1)
        ax.imshow(image, aspect='auto')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

        # Display the label and optionally prediction
        x_label = 'Label: ' + name_map[class_names[label]]
        if prediction is not None:
            x_label = 'Prediction: ' + name_map[class_names[prediction]] + '\n' + x_label
            ax.xaxis.label.set_color('green' if label == prediction else 'red')
        ax.set_xlabel(x_label)

    plt.show()


def preprocess_fn(data):
    image = data['image']

    # Normalize [0, 255] to [0, 1]
    image = tf.cast(image, tf.float32)
    image = image / 255.

    # Resize the images to 224 x 224
    image = tf.image.resize(image, (224, 224))

    data['image'] = image
    return data


batch = dataset['validation'].map(preprocess_fn).batch(25).as_numpy_iterator()
examples = next(batch)
plot(examples)

classifier = hub.KerasLayer('https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2')
probabilities = classifier(examples['image'])
predictions = tf.argmax(probabilities, axis=-1)

plot(examples, predictions)

DATASET = 'cassava'
DATASET_SPLIT = 'test'
BATCH_SIZE = 32
MAX_EXAMPLES = 1000


def label_to_unknown_fn(data):
    data['label'] = 5  # Override label to unknown.
    return data


# Preprocess the examples and map the image label to unknown for non-cassava datasets.
ds = tfds.load(DATASET, split=DATASET_SPLIT).map(preprocess_fn).take(MAX_EXAMPLES)
dataset_description = DATASET
if DATASET != 'cassava':
    ds = ds.map(label_to_unknown_fn)
    dataset_description += ' (labels mapped to unknown)'
ds = ds.batch(BATCH_SIZE)

# Calculate the accuracy of the model
metric = tf.keras.metrics.Accuracy()
for examples in ds:
    probabilities = classifier(examples['image'])
    predictions = tf.math.argmax(probabilities, axis=-1)
    labels = examples['label']
    metric.update_state(labels, predictions)

print('Accuracy on %s: %.2f' % (dataset_description, metric.result().numpy()))
