import os
import time

import numpy as np
import tensorflow as tf
from tflite_model_maker import model_spec
from tflite_model_maker import image_classifier
from tflite_model_maker.config import ExportFormat
from tflite_model_maker.config import QuantizationConfig
from tflite_model_maker.image_classifier import DataLoader
import matplotlib.pyplot as plt

assert tf.__version__.startswith('2')

image_path = tf.keras.utils.get_file(
    'flower_photos.tgz',
    'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',
    extract=True)
image_path = os.path.join(os.path.dirname(image_path), 'flower_photos')

data = DataLoader.from_folder(image_path)
# data = data.gen_dataset(batch_size=1)
train_data, rest_data = data.split(0.8)
# for batch in data.take(1):
#     print(batch)
#     break

validation_data, test_data = rest_data.split(0.5)

inception_v3_spec = image_classifier.ModelSpec(
    uri='https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1')
inception_v3_spec.input_image_shape = [299, 299]
model = image_classifier.create(train_data, validation_data=validation_data,
                                model_spec=inception_v3_spec, epochs=20)

loss, accuracy = model.evaluate(test_data)

config = QuantizationConfig.for_float16()
model.export(export_dir='./testTFlite', tflite_filename='model_fp16.tflite', quantization_config=config, export_format=(ExportFormat.TFLITE, ExportFormat.LABEL))

start = time.time()
print(model.evaluate_tflite('./testTFlite/model_fp16.tflite', test_data))
end = time.time()
print('elapsed time: ', end - start)
