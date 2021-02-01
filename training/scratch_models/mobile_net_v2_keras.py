"""MobileNet v2 models for Keras.

# Reference
- [Inverted Residuals and Linear Bottlenecks Mobile Networks for
   Classification, Detection and Segmentation]
   (https://arxiv.org/abs/1801.04381)
"""

from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout
from keras.layers import Activation, BatchNormalization, add, Reshape
from keras.layers import DepthwiseConv2D
# from keras_applications.mobilenet import relu6
# from keras.applications.mobilenet import relu6
from keras.utils.vis_utils import plot_model
from keras.utils.generic_utils import CustomObjectScope

from keras import backend as K


def relu6(x):
    return K.relu(x, max_value=6)


def _conv_block(inputs, filters, kernel, strides, use_bias=True):
    """Convolution Block
    This function defines a 2D convolution operation with BN and relu6.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        strides: An integer or tuple/list of 2 integers,
            specifying the strides of the convolution along the width and height.
            Can be a single integer to specify the same value for
            all spatial dimensions.

    # Returns
        Output tensor.
    """

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    if nlay < 0 or nlay > 16:
        basename = 'conv_%d' % (nlay + 1)
    else:
        basename = 'expanded_conv_%d_expand' % nlay

    x = Conv2D(filters, kernel, padding='same', strides=strides, name=basename, use_bias=use_bias)(inputs)
    x = BatchNormalization(axis=channel_axis, name=basename + '_batch_normalization')(x)
    return Activation(relu6, name=basename + '_activation')(x)


def _bottleneck(inputs, filters, kernel, t, s, r=False):
    """Bottleneck
    This function defines a basic bottleneck structure.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        r: Boolean, Whether to use the residuals.

    # Returns
        Output tensor.
    """
    global nlay

    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1

    # Create expansions layer only if needed (expansion factor >1)
    if t > 1:
        tchannel = K.int_shape(inputs)[channel_axis] * t
        x = _conv_block(inputs, tchannel, (1, 1), (1, 1), use_bias=False)
    else:
        x = inputs

    x = DepthwiseConv2D(kernel, strides=(s, s), depth_multiplier=1, padding='same',
                        name='expanded_conv_%d_depthwise' % nlay, use_bias=False)(x)
    x = BatchNormalization(axis=channel_axis, name='expanded_conv_%d_depthwise_batch_normalization' % nlay)(x)
    x = Activation(relu6, name='expanded_conv_%d_depthwise_activation' % nlay)(x)

    x = Conv2D(filters, (1, 1), strides=(1, 1), padding='same', name='expanded_conv_%d_project' % nlay, use_bias=False)(
        x)
    x = BatchNormalization(axis=channel_axis, name='expanded_conv_%d_project_batch_normalization' % nlay)(x)

    if r:
        x = add([x, inputs], name="expanded_conv_%d_add" % nlay)

    nlay += 1
    return x


def _inverted_residual_block(inputs, filters, kernel, t, strides, n):
    """Inverted Residual Block
    This function defines a sequence of 1 or more identical layers.

    # Arguments
        inputs: Tensor, input tensor of conv layer.
        filters: Integer, the dimensionality of the output space.
        kernel: An integer or tuple/list of 2 integers, specifying the
            width and height of the 2D convolution window.
        t: Integer, expansion factor.
            t is always applied to the input size.
        s: An integer or tuple/list of 2 integers,specifying the strides
            of the convolution along the width and height.Can be a single
            integer to specify the same value for all spatial dimensions.
        n: Integer, layer repeat times.
    # Returns
        Output tensor.
    """

    x = _bottleneck(inputs, filters, kernel, t, strides)

    for i in range(1, n):
        x = _bottleneck(x, filters, kernel, t, 1, True)

    return x


def roundup(n):
    x = (n + 6) // 8
    return x * 8


def MobileNetv2(input_shape, width_multiplier=1.0, k=0):
    """MobileNetv2
    This function defines a MobileNetv2 architectures.

    # Arguments
        input_shape: An integer or tuple/list of 3 integers, shape
            of input tensor.
        k: Integer, number of classes.
    # Returns
        MobileNetv2 model.
    """

    global nlay
    nlay = -1

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, roundup(int(32 * width_multiplier)), (3, 3), strides=(2, 2), use_bias=False)
    nlay += 1

    fix = 0
    if width_multiplier - 1.3 < 0.01:
        fix = -2

    x = _inverted_residual_block(x, roundup(int(16 * width_multiplier)), (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, roundup(int(24 * width_multiplier)), (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, roundup(int(32 * width_multiplier)), (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, roundup(int(64 * width_multiplier) + fix), (3, 3), t=6, strides=2, n=4)
    x = _inverted_residual_block(x, roundup(int(96 * width_multiplier)), (3, 3), t=6, strides=1, n=3)
    x = _inverted_residual_block(x, roundup(int(160 * width_multiplier)), (3, 3), t=6, strides=2, n=3)
    x = _inverted_residual_block(x, roundup(int(320 * width_multiplier)), (3, 3), t=6, strides=1, n=1)

    last_conv_size = max(1280, int(1280 * width_multiplier))

    x = _conv_block(x, last_conv_size, (1, 1), strides=(1, 1), use_bias=False)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_conv_size), name='reshape_1')(x)
    if k > 0:
        x = Dropout(0.3, name='Dropout')(x)
        x = Conv2D(k, (1, 1), padding='same', name='logits', use_bias=True)(x)

        x = Activation('softmax', name='softmax')(x)
        output = Reshape((k,), name='out')(x)
    else:
        output = x

    model = Model(inputs, output)

    return model


def MobileBioNetSmallerv2(input_shape, width_multiplier=1.0):
    global nlay
    nlay = -1

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, roundup(int(32 * width_multiplier)), (3, 3), strides=(2, 2), use_bias=False)
    nlay += 1

    x = _inverted_residual_block(x, roundup(int(16 * width_multiplier)), (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, roundup(int(24 * width_multiplier)), (3, 3), t=6, strides=2, n=1)
    x = _inverted_residual_block(x, roundup(int(32 * width_multiplier)), (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, roundup(int(64 * width_multiplier)), (3, 3), t=6, strides=1, n=2)

    last_conv_size = max(1280, int(1280 * width_multiplier))

    x = _conv_block(x, last_conv_size, (1, 1), strides=(1, 1), use_bias=False)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_conv_size), name='reshape_1')(x)
    # x = Conv2D(k, (1, 1), padding='same', name='logits', use_bias=True)(x)
    # x = Activation('softmax', name='softmax')(x)
    # x = Reshape((k,), name='out')(x)
    output = x
    model = Model(inputs, output)
    # plot_model(model, to_file='MobileBioNetv2.png', show_shapes=True)
    return model


def MobileBioNetv2(input_shape, width_multiplier=1.0):
    global nlay
    nlay = -1

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, roundup(int(32 * width_multiplier)), (3, 3), strides=(2, 2), use_bias=False)
    nlay += 1

    x = _inverted_residual_block(x, roundup(int(16 * width_multiplier)), (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, roundup(int(24 * width_multiplier)), (3, 3), t=6, strides=2, n=1)
    x = _inverted_residual_block(x, roundup(int(32 * width_multiplier)), (3, 3), t=6, strides=2, n=2)
    x = _inverted_residual_block(x, roundup(int(64 * width_multiplier)), (3, 3), t=6, strides=1, n=4)

    last_conv_size = max(1280, int(1280 * width_multiplier))

    x = _conv_block(x, last_conv_size, (1, 1), strides=(1, 1), use_bias=False)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_conv_size), name='reshape_1')(x)
    # x = Conv2D(k, (1, 1), padding='same', name='logits', use_bias=True)(x)
    # x = Activation('softmax', name='softmax')(x)
    # x = Reshape((k,), name='out')(x)
    output = x
    model = Model(inputs, output)
    # plot_model(model, to_file='MobileBioNetv2.png', show_shapes=True)

    return model


def MobileBioNetSmallestv2(input_shape, width_multiplier=1.0):
    global nlay
    nlay = -1

    inputs = Input(shape=input_shape)
    x = _conv_block(inputs, roundup(int(32 * width_multiplier)), (3, 3), strides=(2, 2), use_bias=False)
    nlay += 1

    x = _inverted_residual_block(x, roundup(int(16 * width_multiplier)), (3, 3), t=1, strides=1, n=1)
    x = _inverted_residual_block(x, roundup(int(24 * width_multiplier)), (3, 3), t=6, strides=2, n=1)
    x = _inverted_residual_block(x, roundup(int(32 * width_multiplier)), (3, 3), t=6, strides=2, n=1)
    x = _inverted_residual_block(x, roundup(int(64 * width_multiplier)), (3, 3), t=6, strides=2, n=1)

    last_conv_size = 1280

    x = _conv_block(x, last_conv_size, (1, 1), strides=(1, 1), use_bias=False)
    x = Dropout(0.2)(x)
    x = GlobalAveragePooling2D()(x)
    x = Reshape((1, 1, last_conv_size), name='reshape_1')(x)
    # x = Conv2D(k, (1, 1), padding='same', name='logits', use_bias=True)(x)
    # x = Activation('softmax', name='softmax')(x)
    # x = Reshape((k,), name='out')(x)
    output = x
    model = Model(inputs, output)
    # plot_model(model, to_file='MobileBioNetv2.png', show_shapes=True)

    return model


if __name__ == '__main__':
    MobileNetv2((224, 224, 3), 100)
