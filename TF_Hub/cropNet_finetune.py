import matplotlib.pyplot as plt
import os
import seaborn as sns

import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow_examples.lite.model_maker.core.export_format import ExportFormat
from tensorflow_examples.lite.model_maker.core.task import image_preprocessing

from tflite_model_maker import image_classifier
from tflite_model_maker import ImageClassifierDataLoader
from tflite_model_maker.image_classifier import ModelSpec

tfds_name = 'cassava'
(ds_train, ds_validation, ds_test), ds_info = tfds.load(
    name=tfds_name,
    split=['train', 'validation', 'test'],
    with_info=True,
    as_supervised=True)
TFLITE_NAME_PREFIX = tfds_name

_ = tfds.show_examples(ds_train, ds_info)

UNKNOWN_TFDS_DATASETS = [{
    'tfds_name': 'imagenet_v2/matched-frequency',
    'train_split': 'test[:80%]',
    'test_split': 'test[80%:]',
    'num_examples_ratio_to_normal': 1.0,
}, {
    'tfds_name': 'oxford_flowers102',
    'train_split': 'train',
    'test_split': 'test',
    'num_examples_ratio_to_normal': 1.0,
}, {
    'tfds_name': 'beans',
    'train_split': 'train',
    'test_split': 'test',
    'num_examples_ratio_to_normal': 1.0,
}]

# Load unknown datasets.
weights = [
    spec['num_examples_ratio_to_normal'] for spec in UNKNOWN_TFDS_DATASETS
]
num_unknown_train_examples = sum(
    int(w * ds_train.cardinality().numpy()) for w in weights)
ds_unknown_train = tf.data.experimental.sample_from_datasets([
    tfds.load(
        name=spec['tfds_name'], split=spec['train_split'],
        as_supervised=True).repeat(-1) for spec in UNKNOWN_TFDS_DATASETS
], weights).take(num_unknown_train_examples)
ds_unknown_train = ds_unknown_train.apply(
    tf.data.experimental.assert_cardinality(num_unknown_train_examples))
ds_unknown_tests = [
    tfds.load(
        name=spec['tfds_name'], split=spec['test_split'], as_supervised=True)
    for spec in UNKNOWN_TFDS_DATASETS
]
ds_unknown_test = ds_unknown_tests[0]
for ds in ds_unknown_tests[1:]:
    ds_unknown_test = ds_unknown_test.concatenate(ds)

# All examples from the unknown datasets will get a new class label number.
num_normal_classes = len(ds_info.features['label'].names)
unknown_label_value = tf.convert_to_tensor(num_normal_classes, tf.int64)
ds_unknown_train = ds_unknown_train.map(lambda image, _:
                                        (image, unknown_label_value))
ds_unknown_test = ds_unknown_test.map(lambda image, _:
                                      (image, unknown_label_value))

# Merge the normal train dataset with the unknown train dataset.
weights = [
    ds_train.cardinality().numpy(),
    ds_unknown_train.cardinality().numpy()
]
ds_train_with_unknown = tf.data.experimental.sample_from_datasets(
    [ds_train, ds_unknown_train], [float(w) for w in weights])
ds_train_with_unknown = ds_train_with_unknown.apply(
    tf.data.experimental.assert_cardinality(sum(weights)))

print((f"Added {ds_unknown_train.cardinality().numpy()} negative examples."
       f"Training dataset has now {ds_train_with_unknown.cardinality().numpy()}"
       ' examples in total.'))


def random_crop_and_random_augmentations_fn(image):
    # preprocess_for_train does random crop and resize internally.
    image = image_preprocessing.preprocess_for_train(image)
    image = tf.image.random_brightness(image, 0.2)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    image = tf.image.random_saturation(image, 0.75, 1.25)
    image = tf.image.random_hue(image, 0.1)
    return image


def random_crop_fn(image):
    # preprocess_for_train does random crop and resize internally.
    image = image_preprocessing.preprocess_for_train(image)
    return image


def resize_and_center_crop_fn(image):
    image = tf.image.resize(image, (256, 256))
    image = image[16:240, 16:240]
    return image


no_augment_fn = lambda image: image

train_augment_fn = lambda image, label: (
    random_crop_and_random_augmentations_fn(image), label)
eval_augment_fn = lambda image, label: (resize_and_center_crop_fn(image), label)

ds_train_with_unknown = ds_train_with_unknown.map(train_augment_fn)
ds_validation = ds_validation.map(eval_augment_fn)
ds_test = ds_test.map(eval_augment_fn)
ds_unknown_test = ds_unknown_test.map(eval_augment_fn)

label_names = ds_info.features['label'].names + ['UNKNOWN']

train_data = ImageClassifierDataLoader(ds_train_with_unknown,
                                       ds_train_with_unknown.cardinality(),
                                       label_names)
validation_data = ImageClassifierDataLoader(ds_validation,
                                            ds_validation.cardinality(),
                                            label_names)
test_data = ImageClassifierDataLoader(ds_test, ds_test.cardinality(),
                                      label_names)
unknown_test_data = ImageClassifierDataLoader(ds_unknown_test,
                                              ds_unknown_test.cardinality(),
                                              label_names)

model_name = 'mobilenet_v3_large_100_224'

map_model_name = {
    'cropnet_cassava':
        'https://tfhub.dev/google/cropnet/feature_vector/cassava_disease_V1/1',
    'cropnet_concat':
        'https://tfhub.dev/google/cropnet/feature_vector/concat/1',
    'cropnet_imagenet':
        'https://tfhub.dev/google/cropnet/feature_vector/imagenet/1',
    'mobilenet_v3_large_100_224':
        'https://tfhub.dev/google/imagenet/mobilenet_v3_large_100_224/feature_vector/5',
}

model_handle = map_model_name[model_name]

image_model_spec = ModelSpec(uri=model_handle)

model = image_classifier.create(
    train_data,
    model_spec=image_model_spec,
    batch_size=64,
    learning_rate=0.03,
    epochs=5,
    shuffle=True,
    train_whole_model=True,
    validation_data=validation_data)

model.evaluate(test_data)


def predict_class_label_number(dataset):
    """Runs inference and returns predictions as class label numbers."""
    rev_label_names = {l: i for i, l in enumerate(label_names)}
    return [
        rev_label_names[o[0][0]]
        for o in model.predict_top_k(dataset, batch_size=128)
    ]


def show_confusion_matrix(cm, labels):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()


confusion_mtx = tf.math.confusion_matrix(
    predict_class_label_number(test_data),
    list(ds_test.map(lambda x, y: y)),
    num_classes=len(label_names))

show_confusion_matrix(confusion_mtx, label_names)

model.evaluate(unknown_test_data)

unknown_confusion_mtx = tf.math.confusion_matrix(
    predict_class_label_number(unknown_test_data),
    list(ds_unknown_test.map(lambda x, y: y)),
    num_classes=len(label_names))

show_confusion_matrix(unknown_confusion_mtx, label_names)

tflite_filename = f'{TFLITE_NAME_PREFIX}_model_{model_name}.tflite'
model.export(export_dir='.', tflite_filename=tflite_filename)

# Export saved model version.
model.export(export_dir='.', export_format=ExportFormat.SAVED_MODEL)
