import numpy as np

from tensorflow.python.eager import context
from tensorflow.python.compat import compat
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.keras import backend
from tensorflow.python.keras.engine import base_preprocessing_layer
from tensorflow.python.keras.engine.base_preprocessing_layer import PreprocessingLayer
from tensorflow.python.keras.engine.input_spec import InputSpec
from tensorflow.python.keras.utils import control_flow_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_image_ops
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import stateful_random_ops
from tensorflow.python.ops import stateless_random_ops
from tensorflow.python.util.tf_export import keras_export
from tensorflow.keras.layers.experimental.preprocessing import *
import tensorflow as tf

ResizeMethod = image_ops.ResizeMethod

_RESIZE_METHODS = {
    'bilinear': ResizeMethod.BILINEAR,
    'nearest': ResizeMethod.NEAREST_NEIGHBOR,
    'bicubic': ResizeMethod.BICUBIC,
    'area': ResizeMethod.AREA,
    'lanczos3': ResizeMethod.LANCZOS3,
    'lanczos5': ResizeMethod.LANCZOS5,
    'gaussian': ResizeMethod.GAUSSIAN,
    'mitchellcubic': ResizeMethod.MITCHELLCUBIC
}

H_AXIS = 1
W_AXIS = 2


def check_fill_mode_and_interpolation(fill_mode, interpolation):
    if fill_mode not in {'reflect', 'wrap', 'constant', 'nearest'}:
        raise NotImplementedError(
            'Unknown `fill_mode` {}. Only `reflect`, `wrap`, '
            '`constant` and `nearest` are supported.'.format(fill_mode))
    if interpolation not in {'nearest', 'bilinear'}:
        raise NotImplementedError('Unknown `interpolation` {}. Only `nearest` and '
                                  '`bilinear` are supported.'.format(interpolation))


def get_rotation_matrix(angles, image_height, image_width, name=None):
    """Returns projective transform(s) for the given angle(s).

  Args:
    angles: A scalar angle to rotate all images by, or (for batches of images) a
      vector with an angle to rotate each image in the batch. The rank must be
      statically known (the shape is not `TensorShape(None)`).
    image_height: Height of the image(s) to be transformed.
    image_width: Width of the image(s) to be transformed.
    name: The name of the op.

  Returns:
    A tensor of shape (num_images, 8). Projective transforms which can be given
      to operation `image_projective_transform_v2`. If one row of transforms is
       [a0, a1, a2, b0, b1, b2, c0, c1], then it maps the *output* point
       `(x, y)` to a transformed *input* point
       `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`,
       where `k = c0 x + c1 y + 1`.
  """
    with backend.name_scope(name or 'rotation_matrix'):
        x_offset = ((image_width - 1) - (math_ops.cos(angles) *
                                         (image_width - 1) - math_ops.sin(angles) *
                                         (image_height - 1))) / 2.0
        y_offset = ((image_height - 1) - (math_ops.sin(angles) *
                                          (image_width - 1) + math_ops.cos(angles) *
                                          (image_height - 1))) / 2.0
        num_angles = array_ops.shape(angles)[0]
        return array_ops.concat(
            values=[
                math_ops.cos(angles)[:, None],
                -math_ops.sin(angles)[:, None],
                x_offset[:, None],
                math_ops.sin(angles)[:, None],
                math_ops.cos(angles)[:, None],
                y_offset[:, None],
                array_ops.zeros((num_angles, 2), dtypes.float32),
            ],
            axis=1)


def get_translation_matrix(translations, name=None):
    """Returns projective transform(s) for the given translation(s).

  Args:
    translations: A matrix of 2-element lists representing [dx, dy] to translate
      for each image (for a batch of images).
    name: The name of the op.

  Returns:
    A tensor of shape (num_images, 8) projective transforms which can be given
      to `transform`.
  """
    with backend.name_scope(name or 'translation_matrix'):
        num_translations = array_ops.shape(translations)[0]
        # The translation matrix looks like:
        #     [[1 0 -dx]
        #      [0 1 -dy]
        #      [0 0 1]]
        # where the last entry is implicit.
        # Translation matrices are always float32.
        return array_ops.concat(
            values=[
                array_ops.ones((num_translations, 1), dtypes.float32),
                array_ops.zeros((num_translations, 1), dtypes.float32),
                -translations[:, 0, None],
                array_ops.zeros((num_translations, 1), dtypes.float32),
                array_ops.ones((num_translations, 1), dtypes.float32),
                -translations[:, 1, None],
                array_ops.zeros((num_translations, 2), dtypes.float32),
            ],
            axis=1)


def transform(images,
              transforms,
              fill_mode='reflect',
              fill_value=0.0,
              interpolation='bilinear',
              output_shape=None,
              name=None):
    """Applies the given transform(s) to the image(s).

  Args:
    images: A tensor of shape (num_images, num_rows, num_columns, num_channels)
      (NHWC), (num_rows, num_columns, num_channels) (HWC), or (num_rows,
      num_columns) (HW). The rank must be statically known (the shape is not
      `TensorShape(None)`.
    transforms: Projective transform matrix/matrices. A vector of length 8 or
      tensor of size N x 8. If one row of transforms is [a0, a1, a2, b0, b1, b2,
      c0, c1], then it maps the *output* point `(x, y)` to a transformed *input*
      point `(x', y') = ((a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k)`, where
      `k = c0 x + c1 y + 1`. The transforms are *inverted* compared to the
      transform mapping input points to output points. Note that gradients are
      not backpropagated into transformation parameters.
    fill_mode: Points outside the boundaries of the input are filled according
      to the given mode (one of `{'constant', 'reflect', 'wrap', 'nearest'}`).
    fill_value: a float represents the value to be filled outside the boundaries
      when `fill_mode` is "constant".
    interpolation: Interpolation mode. Supported values: "nearest", "bilinear".
    output_shape: Output dimesion after the transform, [height, width]. If None,
      output is the same size as input image.
    name: The name of the op.  ## Fill mode.
  Behavior for each valid value is as follows:  reflect (d c b a | a b c d | d c
    b a) The input is extended by reflecting about the edge of the last pixel.
    constant (k k k k | a b c d | k k k k) The input is extended by filling all
    values beyond the edge with the same constant value k = 0.  wrap (a b c d |
    a b c d | a b c d) The input is extended by wrapping around to the opposite
    edge.  nearest (a a a a | a b c d | d d d d) The input is extended by the
    nearest pixel.
  Input shape:
    4D tensor with shape: `(samples, height, width, channels)`,
      data_format='channels_last'.
  Output shape:
    4D tensor with shape: `(samples, height, width, channels)`,
      data_format='channels_last'.

  Returns:
    Image(s) with the same type and shape as `images`, with the given
    transform(s) applied. Transformed coordinates outside of the input image
    will be filled with zeros.

  Raises:
    TypeError: If `image` is an invalid type.
    ValueError: If output shape is not 1-D int32 Tensor.
  """
    with backend.name_scope(name or 'transform'):
        if output_shape is None:
            output_shape = array_ops.shape(images)[1:3]
            if not context.executing_eagerly():
                output_shape_value = tensor_util.constant_value(output_shape)
                if output_shape_value is not None:
                    output_shape = output_shape_value

        output_shape = ops.convert_to_tensor_v2_with_dispatch(
            output_shape, dtypes.int32, name='output_shape')

        if not output_shape.get_shape().is_compatible_with([2]):
            raise ValueError('output_shape must be a 1-D Tensor of 2 elements: '
                             'new_height, new_width, instead got '
                             '{}'.format(output_shape))

        fill_value = ops.convert_to_tensor_v2_with_dispatch(
            fill_value, dtypes.float32, name='fill_value')

        if compat.forward_compatible(2020, 8, 5):
            return gen_image_ops.ImageProjectiveTransformV3(
                images=images,
                output_shape=output_shape,
                fill_value=fill_value,
                transforms=transforms,
                fill_mode=fill_mode.upper(),
                interpolation=interpolation.upper())

        return gen_image_ops.ImageProjectiveTransformV2(
            images=images,
            output_shape=output_shape,
            transforms=transforms,
            fill_mode=fill_mode.upper(),
            interpolation=interpolation.upper())


def make_generator(seed=None):
    """Creates a random generator.

  Args:
    seed: the seed to initialize the generator. If None, the generator will be
      initialized non-deterministically.

  Returns:
    A generator object.
  """
    if seed:
        return stateful_random_ops.Generator.from_seed(seed)
    else:
        return stateful_random_ops.Generator.from_non_deterministic_state()


HORIZONTAL = 'horizontal'
VERTICAL = 'vertical'
HORIZONTAL_AND_VERTICAL = 'horizontal_and_vertical'


class RandomFlip_prob(RandomFlip):
    def __init__(self,
                 mode=HORIZONTAL_AND_VERTICAL,
                 seed=None,
                 p=0.5,
                 **kwargs):
        super(RandomFlip, self).__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomFlip').set(True)
        self.mode = mode
        self.p = p
        if mode == HORIZONTAL:
            self.horizontal = True
            self.vertical = False
        elif mode == VERTICAL:
            self.horizontal = False
            self.vertical = True
        elif mode == HORIZONTAL_AND_VERTICAL:
            self.horizontal = True
            self.vertical = True
        else:
            raise ValueError('RandomFlip layer {name} received an unknown mode '
                             'argument {arg}'.format(name=self.name, arg=mode))
        self.seed = seed
        self._rng = make_generator(self.seed)
        self.input_spec = InputSpec(ndim=4)

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        def random_flipped_inputs():
            flipped_outputs = inputs
            if tf.random.uniform([]) < self.p:
                return flipped_outputs
            if self.horizontal:
                flipped_outputs = image_ops.random_flip_left_right(
                    flipped_outputs, self.seed)
            if self.vertical:
                flipped_outputs = image_ops.random_flip_up_down(flipped_outputs,
                                                                self.seed)
            return flipped_outputs

        output = control_flow_util.smart_cond(training, random_flipped_inputs,
                                              lambda: inputs)
        output.set_shape(inputs.shape)
        return output

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = {
            'mode': self.mode,
            'seed': self.seed,
        }
        base_config = super(RandomFlip, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class RandomTranslation_prob(RandomTranslation):
    def __init__(self,
                 height_factor,
                 width_factor,
                 fill_mode='reflect',
                 interpolation='bilinear',
                 seed=None,
                 p=0.5,
                 fill_value=0.0,
                 **kwargs):
        self.height_factor = height_factor
        self.p = p
        if isinstance(height_factor, (tuple, list)):
            self.height_lower = height_factor[0]
            self.height_upper = height_factor[1]
        else:
            self.height_lower = -height_factor
            self.height_upper = height_factor
        if self.height_upper < self.height_lower:
            raise ValueError('`height_factor` cannot have upper bound less than '
                             'lower bound, got {}'.format(height_factor))
        if abs(self.height_lower) > 1. or abs(self.height_upper) > 1.:
            raise ValueError('`height_factor` must have values between [-1, 1], '
                             'got {}'.format(height_factor))

        self.width_factor = width_factor
        if isinstance(width_factor, (tuple, list)):
            self.width_lower = width_factor[0]
            self.width_upper = width_factor[1]
        else:
            self.width_lower = -width_factor
            self.width_upper = width_factor
        if self.width_upper < self.width_lower:
            raise ValueError('`width_factor` cannot have upper bound less than '
                             'lower bound, got {}'.format(width_factor))
        if abs(self.width_lower) > 1. or abs(self.width_upper) > 1.:
            raise ValueError('`width_factor` must have values between [-1, 1], '
                             'got {}'.format(width_factor))

        check_fill_mode_and_interpolation(fill_mode, interpolation)

        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self._rng = make_generator(self.seed)
        self.input_spec = InputSpec(ndim=4)
        super(RandomTranslation, self).__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomTranslation').set(
            True)

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        def random_translated_inputs():
            if tf.random.uniform([]) < self.p:
                return inputs
            """Translated inputs with random ops."""
            inputs_shape = array_ops.shape(inputs)
            batch_size = inputs_shape[0]
            h_axis, w_axis = H_AXIS, W_AXIS
            img_hd = math_ops.cast(inputs_shape[h_axis], dtypes.float32)
            img_wd = math_ops.cast(inputs_shape[w_axis], dtypes.float32)
            height_translate = self._rng.uniform(
                shape=[batch_size, 1],
                minval=self.height_lower,
                maxval=self.height_upper,
                dtype=dtypes.float32)
            height_translate = height_translate * img_hd
            width_translate = self._rng.uniform(
                shape=[batch_size, 1],
                minval=self.width_lower,
                maxval=self.width_upper,
                dtype=dtypes.float32)
            width_translate = width_translate * img_wd
            translations = math_ops.cast(
                array_ops.concat([width_translate, height_translate], axis=1),
                dtype=dtypes.float32)
            return transform(
                inputs,
                get_translation_matrix(translations),
                interpolation=self.interpolation,
                fill_mode=self.fill_mode,
                fill_value=self.fill_value)

        output = control_flow_util.smart_cond(training, random_translated_inputs,
                                              lambda: inputs)
        output.set_shape(inputs.shape)
        return output


class RandomRotation_prob(RandomRotation):
    def __init__(self,
                 factor,
                 fill_mode='reflect',
                 interpolation='bilinear',
                 seed=None,
                 p=0.5,
                 fill_value=0.0,
                 **kwargs):
        self.factor = factor
        self.p = p
        if isinstance(factor, (tuple, list)):
            self.lower = factor[0]
            self.upper = factor[1]
        else:
            self.lower = -factor
            self.upper = factor
        if self.upper < self.lower:
            raise ValueError('Factor cannot have negative values, '
                             'got {}'.format(factor))
        check_fill_mode_and_interpolation(fill_mode, interpolation)
        self.fill_mode = fill_mode
        self.fill_value = fill_value
        self.interpolation = interpolation
        self.seed = seed
        self._rng = make_generator(self.seed)
        self.input_spec = InputSpec(ndim=4)
        super(RandomRotation, self).__init__(**kwargs)
        base_preprocessing_layer.keras_kpl_gauge.get_cell('RandomRotation').set(
            True)

    def call(self, inputs, training=True):
        if training is None:
            training = backend.learning_phase()

        def random_rotated_inputs():
            if tf.random.uniform([]) < self.p:
                return inputs
            """Rotated inputs with random ops."""
            inputs_shape = array_ops.shape(inputs)
            batch_size = inputs_shape[0]
            img_hd = math_ops.cast(inputs_shape[H_AXIS], dtypes.float32)
            img_wd = math_ops.cast(inputs_shape[W_AXIS], dtypes.float32)
            min_angle = self.lower * 2. * np.pi
            max_angle = self.upper * 2. * np.pi
            angles = self._rng.uniform(
                shape=[batch_size], minval=min_angle, maxval=max_angle)
            return transform(
                inputs,
                get_rotation_matrix(angles, img_hd, img_wd),
                fill_mode=self.fill_mode,
                fill_value=self.fill_value,
                interpolation=self.interpolation)

        output = control_flow_util.smart_cond(training, random_rotated_inputs,
                                              lambda: inputs)
        output.set_shape(inputs.shape)
        return output
