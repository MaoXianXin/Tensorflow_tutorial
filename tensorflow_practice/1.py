import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers
import tensorflow_datasets as tfds
from imgaug import augmenters as iaa
import imgaug as ia

tfds.disable_progress_bar()
tf.random.set_seed(42)
ia.seed(42)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

AUTO = tf.data.AUTOTUNE
BATCH_SIZE = 128
EPOCHS = 30
IMAGE_SIZE = 72

rand_aug = iaa.RandAugment(n=3, m=7)


def augment(images):
    # Input to `augment()` is a TensorFlow tensor which
    # is not supported by `imgaug`. This is why we first
    # convert it to its `numpy` variant.
    images = tf.cast(images, tf.uint8)
    return rand_aug(images=images.numpy())


train_ds_rand = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
        .cache()
        .map(
        lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y),
        num_parallel_calls=AUTO,
    )
        .map(
        lambda x, y: (tf.py_function(augment, [x], [tf.float32])[0], y),
        num_parallel_calls=AUTO,
    )
        .prefetch(AUTO)
)

test_ds = (
    tf.data.Dataset.from_tensor_slices((x_test, y_test))
        .batch(BATCH_SIZE)
        .cache()
        .map(
        lambda x, y: (tf.image.resize(x, (IMAGE_SIZE, IMAGE_SIZE)), y),
        num_parallel_calls=AUTO,
    )
        .prefetch(AUTO)
)

simple_aug = tf.keras.Sequential([
    layers.experimental.preprocessing.Resizing(IMAGE_SIZE, IMAGE_SIZE),
    layers.experimental.preprocessing.RandomFlip('horizontal'),
    layers.experimental.preprocessing.RandomRotation(factor=0.02),
    layers.experimental.preprocessing.RandomZoom(
        height_factor=0.2, width_factor=0.2
    )
])

train_ds_simple = (
    tf.data.Dataset.from_tensor_slices((x_train, y_train))
        .shuffle(BATCH_SIZE * 100)
        .batch(BATCH_SIZE)
        .cache()
        .map(lambda x, y: (simple_aug(x), y), num_parallel_calls=AUTO)
        .prefetch(AUTO)
)

sample_images, _ = next(iter(train_ds_rand))
plt.figure(figsize=(10, 10))
for i, image in enumerate(sample_images[:9]):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("int"))
    plt.axis("off")
plt.show()

sample_images, _ = next(iter(train_ds_simple))
plt.figure(figsize=(10, 10))
for i, image in enumerate(sample_images[:9]):
    ax = plt.subplot(3, 3, i + 1)
    plt.imshow(image.numpy().astype("int"))
    plt.axis("off")
plt.show()


def get_training_model():
    resnet50_v2 = tf.keras.applications.ResNet50V2(
        weights=None,
        include_top=True,
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        classes=10
    )
    model = tf.keras.Sequential([
        layers.InputLayer((IMAGE_SIZE, IMAGE_SIZE, 3)),
        layers.experimental.preprocessing.Rescaling(scale=1.0 / 127.5, offset=-1),
        resnet50_v2
    ])

    return model


print(get_training_model().summary())

initial_model = get_training_model()
initial_model.save_weights("initial_weights.h5")

rand_aug_model = get_training_model()
rand_aug_model.load_weights("initial_weights.h5")
rand_aug_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
rand_aug_model.fit(train_ds_rand, validation_data=test_ds, epochs=EPOCHS)
_, test_acc = rand_aug_model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_acc * 100))

simple_aug_model = get_training_model()
simple_aug_model.load_weights("initial_weights.h5")
simple_aug_model.compile(
    loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
)
simple_aug_model.fit(train_ds_simple, validation_data=test_ds, epochs=EPOCHS)
_, test_acc = simple_aug_model.evaluate(test_ds)
print("Test accuracy: {:.2f}%".format(test_acc * 100))
