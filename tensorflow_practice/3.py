import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
import datetime
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.python.keras import backend
from tensorflow.python.platform import tf_logging as logging
from custom_augmentation import *

import pathlib

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--key", type=str)
args = parser.parse_args()

batch_size = 128
img_height = 180
img_width = 180
img_size = (img_height, img_width, 3)

augmentation_dict = {
    'RandomFlip': tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
    'RandomRotation': tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
    'RandomContrast': tf.keras.layers.experimental.preprocessing.RandomContrast(0.2),
    'RandomZoom': tf.keras.layers.experimental.preprocessing.RandomZoom(height_factor=0.1, width_factor=0.1),
    'RandomTranslation': tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=0.1, width_factor=0.1),
    'RandomCrop': tf.keras.layers.experimental.preprocessing.RandomCrop(img_height, img_width),
    'RandomFlip_prob': RandomFlip_prob("horizontal_and_vertical"),
    'RandomRotation_prob': RandomRotation_prob(0.2),
    'RandomTranslation_prob': RandomTranslation_prob(height_factor=0.1, width_factor=0.1),
}

dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
data_dir = tf.keras.utils.get_file(origin=dataset_url,
                                   fname='flower_photos',
                                   untar=True)
data_dir = pathlib.Path(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

AUTOTUNE = tf.data.AUTOTUNE

train_ds = train_ds.shuffle(buffer_size=1000).cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

num_classes = 5

data_augmentation = tf.keras.Sequential([
    augmentation_dict[args.key],
])

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
base_model = tf.keras.applications.MobileNetV2(input_shape=img_size,
                                               include_top=False,
                                               weights='imagenet')
base_model.trainable = False

inputs = tf.keras.Input(shape=img_size)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes)(x)
model = tf.keras.Model(inputs, outputs)
print(model.summary())

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(
    optimizer=optimizer,
    loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy'])

log_dir = "logs/fit_2/mobilenetv2_" + str(args.key) + '_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(log_dir + '/lr')
file_writer.set_as_default()
early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=5,
                                              restore_best_weights=True)
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


class MyCallback(ReduceLROnPlateau):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = backend.get_value(self.model.optimizer.lr)
        current = logs.get(self.monitor)
        if current is None:
            logging.warning('Learning rate reduction is conditioned on metric `%s` '
                            'which is not available. Available metrics are: %s',
                            self.monitor, ','.join(list(logs.keys())))

        else:
            if self.in_cooldown():
                self.cooldown_counter -= 1
                self.wait = 0

            if self.monitor_op(current, self.best):
                self.best = current
                self.wait = 0
            elif not self.in_cooldown():
                self.wait += 1
                if self.wait >= self.patience:
                    old_lr = backend.get_value(self.model.optimizer.lr)
                    if old_lr > np.float32(self.min_lr):
                        new_lr = old_lr * self.factor
                        new_lr = max(new_lr, self.min_lr)
                        tf.summary.scalar('learning rate', data=new_lr, step=epoch)
                        backend.set_value(self.model.optimizer.lr, new_lr)
                        if self.verbose > 0:
                            print('\nEpoch %05d: ReduceLROnPlateau reducing learning '
                                  'rate to %s.' % (epoch + 1, new_lr))
                        self.cooldown_counter = self.cooldown
                        self.wait = 0


reduce_lr = MyCallback(monitor='val_loss', factor=0.2,
                       patience=3, min_lr=1e-6)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=100,
    callbacks=[reduce_lr, early_stop, tensorboard_callback],
    verbose=2
)

print(model.evaluate(val_ds))
