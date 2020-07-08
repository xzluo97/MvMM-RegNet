# -*- coding: utf-8 -*-
"""
Auxiliary functions and operations for network construction, some of which have
been deprecated for high-level modules in TensorFlow.

@author: Xinzhe Luo
"""

from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
import numpy as np
from core.utils_2d import transform, resize, affine_to_shift, get_reference_grid_numpy


#######################################################################
# Some low-level or deprecated implementations of network layers
#######################################################################

def weight_variable(shape, name="weight"):
    fan_in, fan_out = shape[-2:]
    low = -1 * np.sqrt(6.0 / (fan_in + fan_out))  # use 4 for sigmoid, 1 for tanh activation
    high = 1 * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32), name=name)


def weight_variable_devonc(shape, name="weight_deconv"):
    fan_in, fan_out = shape[-2:]
    low = -1 * np.sqrt(6.0 / (fan_in + fan_out))  # use 4 for sigmoid, 1 for tanh activation
    high = 1 * np.sqrt(6.0 / (fan_in + fan_out))
    return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high, dtype=tf.float32), name=name)


def bias_variable(shape, name="bias"):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)


def conv3d(x, W, keep_prob_):
    conv_3d = tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')
    return tf.nn.dropout(conv_3d, keep_prob_)


def deconv3d(x, w, stride):
    x_shape = tf.shape(x)
    output_shape = tf.stack([x_shape[0], x_shape[1] * 2, x_shape[2] * 2, x_shape[3] * 2, x_shape[4] // 2])
    return tf.nn.conv3d_transpose(x, w, output_shape, strides=[1, stride, stride, stride, 1], padding='VALID')


def max_pool3d(x, n):
    return tf.nn.max_pool3d(x, ksize=[1, n, n, n, 1], strides=[1, n, n, n, 1], padding='VALID')


'''
def batch_norm(x, train_phase):
    x_norm = tf.layers.batch_normalization(x, axis=0, training=train_phase)
    return x_norm
'''


def pixel_wise_softmax(output_map):
    """
    deprecated function for tf.nn.softmax
    """
    exponential_map = tf.exp(output_map)
    evidence = tf.add(exponential_map, tf.reverse(exponential_map, [False, False, False, True]))
    return tf.divide(exponential_map, evidence, name="pixel_wise_softmax")


def pixel_wise_softmax_2(output_map):
    """
    deprecated function for tf.nn.softmax
    """
    exponential_map = tf.exp(output_map)
    sum_exp = tf.reduce_sum(exponential_map, -1, keepdims=True)
    tensor_sum_exp = tf.tile(sum_exp, tf.stack([1, 1, 1, 1, tf.shape(output_map)[-1]]))
    return tf.clip_by_value(tf.divide(exponential_map, tensor_sum_exp), 1e-10, 1.0)


def cross_entropy_map(labels, probs):
    """
    Compute the element-wise cross-entropy map by clipping the values of softmax probabilities to avoid Nan loss.

    :param labels: ground-truth value using one-hot representation
    :param probs: probability map as the output of softmax
    :return: A tensor of the same shape as lables and of the same shape as probs with the cross entropy loss.
    """
    return tf.reduce_sum(- labels * tf.log(tf.clip_by_value(probs, 1e-7, 1.0)), axis=-1, name="cross_entropy_map")


def balance_weight_map(flat_labels):
    """
    :param flat_labels: masked ground truth tensor in shape [-1, n_class]
    :return the balance weight map in 1-D tensor
    """
    n = tf.shape(flat_labels)[0]
    return tf.reduce_sum(tf.multiply(flat_labels, tf.tile(1 / tf.reduce_sum(flat_labels, axis=0, keepdims=True),
                                                          [n, 1])), axis=-1, name='balance_weight_map')


#######################################################################
# Tensor manipulations
#######################################################################

def crop_and_concat(x1, x2):
    """
    Crop x1 to match the size of x2 and concatenate them.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2,
               0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size, name='crop')
    crop_concat = tf.concat([x1_crop, x2], -1, name='crop_concat')
    crop_concat.set_shape([None, None, None, None, x1.get_shape().as_list()[-1] + x2.get_shape().as_list()[-1]])
    return crop_concat


def crop_and_add(x1, x2):
    """
    Crop x1 to match the size of x2 and add them together.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2,
               0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size, name='crop')
    return tf.add(x1_crop, x2, name='crop_add')


def pad_and_concat(x1, x2):
    """
    Pad x2 to match the size of x1 and concatenate them.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2,
               0]
    paddings = [[0, 0],
                [offsets[1], x1_shape[1] - x2_shape[1] - offsets[1]],
                [offsets[2], x1_shape[2] - x2_shape[2] - offsets[2]],
                [offsets[3], x1_shape[3] - x2_shape[3] - offsets[3]],
                [0, 0]]
    x2_pad = tf.pad(x2, paddings, name='pad')
    pad_concat = tf.concat([x1, x2_pad], -1, name='pad_concat')
    pad_concat.set_shape([None, None, None, None, x1.get_shape().as_list()[-1] + x2.get_shape().as_list()[-1]])
    return pad_concat


def pad_and_add(x1, x2):
    """
    Pad x2 to match the size of x1 and add them together.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2,
               0]
    paddings = [[0, 0],
                [offsets[1], x1_shape[1] - x2_shape[1] - offsets[1]],
                [offsets[2], x1_shape[2] - x2_shape[2] - offsets[2]],
                [offsets[3], x1_shape[3] - x2_shape[3] - offsets[3]],
                [0, 0]]
    x2_pad = tf.pad(x2, paddings, name='pad')
    return tf.add(x1, x2_pad, name='pad_add')


def crop_to_tensor(x1, x2):
    """
    Crop tensor x1 to match the shape of x2.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    # offsets for the top left corner of the crop
    offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, (x1_shape[3] - x2_shape[3]) // 2,
               0]
    size = [-1, x2_shape[1], x2_shape[2], x2_shape[3], -1]
    x1_crop = tf.slice(x1, offsets, size, name='crop')
    x1_crop.set_shape([None, None, None, None, x1.get_shape().as_list()[-1]])
    return x1_crop


def pad_to_tensor(x1, x2):
    """
    Pad tensor x1 to match the shape of x2.
    """
    x1_shape = tf.shape(x1)
    x2_shape = tf.shape(x2)
    offsets = [0, (x2_shape[1] - x1_shape[1]) // 2, (x2_shape[2] - x1_shape[2]) // 2, (x2_shape[3] - x1_shape[3]) // 2,
               0]
    paddings = [[0, 0],
                [offsets[1], x2_shape[1] - x1_shape[1] - offsets[1]],
                [offsets[2], x2_shape[2] - x1_shape[2] - offsets[2]],
                [offsets[3], x2_shape[3] - x1_shape[3] - offsets[3]],
                [0, 0]]

    x1_pad = tf.pad(x1, paddings, name='pad')

    return x1_pad


#######################################################################
# Pre-defined and integrated implementation of network layers
#######################################################################

def transition_block_layer(inputs, pool_size, filter_size, dropout_rate=0., compression_rate=1,
                           initializer=tf.initializers.he_uniform(), normalizer=None, regularizer=None,
                           train_phase=True, trainable=True, name_or_scope='transition_block', **kwargs):
    """
    Apply a transition block composed of a 1x1 convolution and a max-pooling layer to the given inputs.

    :param inputs: Input feature maps.
    :param pool_size: Size of the pooling window.
    :param filter_size: The size of the convolution kernel.
    :param compression_rate: The compression factor for model compactness in the transition block, set to 1. if without
        compression, value set > 1. for expansion.
    :param initializer: weight initializer, default as Kaiming uniform initialization
    :param normalizer: type of normalization to use, default is None,
        choose from None, 'batch', 'group', 'layer', 'instance', 'batch_instance'
    :param regularizer: Regularizer for weights.
    :param train_phase: Whether in training or in inference mode.
    :param trainable: Whether add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES.
    :param dropout_rate: Dropout probability.
    :param name_or_scope: The scope to open.
    :return: Down-sampled feature maps.
    """
    dropout_type = kwargs.pop('dropout_type', 'regular')
    with tf.variable_scope(name_or_scope):
        regularization_loss = 0.
        input_feature_size = inputs.get_shape().as_list()[-1]
        with tf.variable_scope('transition_block_layer'):
            pool = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_size, name='max_pool')(inputs)
            conv2d = tf.keras.layers.Conv2D(filters=int(compression_rate * input_feature_size),
                                            kernel_size=filter_size, padding='same', use_bias=False,
                                            kernel_initializer=initializer, kernel_regularizer=regularizer,
                                            trainable=trainable, name='conv')
            conv = conv2d(pool)
            if regularizer == 'l2':
                regularization_loss += tf.reduce_sum(tf.square(conv2d.kernel))
            if regularizer == 'l1':
                regularization_loss += tf.reduce_sum(tf.abs(conv2d.kernel))
            norm, loss = normalize(conv, type=normalizer, training=train_phase, regularizer=regularizer, **kwargs)
            regularization_loss += loss
            relu = tf.nn.relu(norm, name='relu')
            feature_maps = dropout_layer(relu, dropout_rate, train_phase, type=dropout_type)
    return feature_maps, regularization_loss


def residual_block_layer(inputs, filter_size, feature_size, num_layers=2, strides=1,
                         dilation_rate=1, padding='same', dropout_rate=0.,
                         initializer=tf.initializers.he_uniform(), normalizer=None, regularizer=None,
                         train_phase=True, trainable=True, name_or_scope='residual_block', **kwargs):
    """
    Apply residual block layers to the given inputs.

    :param inputs: Input feature maps.]
    :param filter_size: Size of the convolution kernel.
    :param feature_size: The number of filters in each convolution.
    :param num_layers: Number of convolutional block layers.
    :param strides: The strides of the convolution.
    :param dilation_rate: the dilation rate to use for dilated convolution.
    :param initializer: weight initializer, default as Kaiming uniform initialization
    :param normalizer: type of normalization to use, default is None,
        choose from None, 'batch', 'group', 'layer', 'instance', 'batch_instance'
    :param regularizer: Regularizer for weights.
    :param padding: The padding type, one of "valid" or "same".
    :param train_phase: Whether in training or in inference mode.
    :param trainable: Whether add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES.
    :param res_kernel_size: The residual block kernel size.
    :param dropout_rate: Dropout probability.
    :param name_or_scope: The scope to open.
    :return: Feature maps of shape [None, None, None, None, feature_size].
    """
    dropout_type = kwargs.pop('dropout_type', 'regular')
    with tf.variable_scope(name_or_scope):
        assert inputs.get_shape().as_list()[-1] == feature_size, "The number of input feature maps must be equal to the" \
                                                                 " output feature maps in the residual block."
        regularization_loss = 0.
        feature_maps = inputs
        for layer in range(num_layers):
            with tf.variable_scope('res_block_layer%s' % layer):
                conv2d = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=filter_size, strides=strides,
                                                padding=padding, dilation_rate=dilation_rate, use_bias=False,
                                                kernel_initializer=initializer,
                                                kernel_regularizer=regularizer, trainable=trainable,
                                                name='conv')
                conv = conv2d(feature_maps)
                if regularizer == 'l2':
                    regularization_loss += tf.reduce_sum(tf.square(conv2d.kernel))
                if regularizer == 'l1':
                    regularization_loss += tf.reduce_sum(tf.abs(conv2d.kernel))
                norm, loss = normalize(conv, type=normalizer, training=train_phase, regularizer=regularizer, **kwargs)
                regularization_loss += loss
                if layer < num_layers - 1:
                    relu = tf.nn.relu(norm, name='relu')
                    feature_maps = dropout_layer(relu, dropout_rate, train_phase, type=dropout_type)
                else:
                    feature_maps = norm

        # res_conv = tf.layers.conv3d(inputs, filters=feature_size,
        #                             kernel_size=res_kernel_size, padding=padding, use_bias=False,
        #                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
        #                             kernel_regularizer=regularizer, trainable=trainable,
        #                             name='res_conv')
        # res_norm = normalize(res_conv, type=normalizer, **kwargs)
        # res_relu = tf.nn.relu(res_norm, name='res_relu')

        res_add = tf.add(feature_maps, inputs, name='res_add')
        relu = tf.nn.relu(res_add, name='output_feature_maps')
        outputs = dropout_layer(relu, dropout_rate, train_phase, type=dropout_type)

        # alternative
        # outputs = tf.add(tf.nn.relu(bn1), inputs, name='output_feature_maps')

    return outputs, regularization_loss


def conv_block_layer(inputs, filter_size, feature_size, num_layers=2, strides=1, padding='same',
                     dilation_rate=1, dropout_rate=0., initializer=tf.initializers.he_uniform(),
                     normalizer=None, regularizer=None, train_phase=True, trainable=True,
                     name_or_scope='conv_block', **kwargs):
    """
    Apply convolutional block layers to the given inputs.

    :param inputs: Input feature maps.
    :param num_layers: Number of convolutional block layers.
    :param filter_size: Size of the convolution kernel.
    :param feature_size: The number of filters in each convolution.
    :param strides: The strides of the convolution.
    :param dilation_rate: the dilation rate to use for dilated convolution.
    :param initializer: weight initializer, default as Kaiming uniform initialization for ReLU activation
    :param normalizer: type of normalization to use, default is None,
        choose from None, 'batch', 'group', 'layer', 'instance', 'batch_instance'
    :param regularizer: The regularizer for weights.
    :param padding: The padding type, one of "valid" or "same".
    :param train_phase: Whether in training or in inference mode.
    :param trainable: Whether add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES.
    :param dropout_rate: Dropout probability.
    :param name_or_scope: The scope to open.
    :return: Feature maps of shape [None, None, None, None, feature_size]
    """
    dropout_type = kwargs.pop('dropout_type', 'regular')
    with tf.variable_scope(name_or_scope):
        regularization_loss = 0.
        feature_maps = inputs
        for k in range(num_layers):
            with tf.variable_scope('conv_block_layer%d' % k):
                conv2d = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=filter_size, strides=strides,
                                                padding=padding, dilation_rate=dilation_rate, use_bias=False,
                                                kernel_initializer=initializer,
                                                kernel_regularizer=regularizer, trainable=trainable,
                                                name='conv')
                conv = conv2d(feature_maps)
                if regularizer == 'l2':
                    regularization_loss += tf.reduce_sum(tf.square(conv2d.kernel))
                if regularizer == 'l1':
                    regularization_loss += tf.reduce_sum(tf.abs(conv2d.kernel))
                norm, loss = normalize(conv, type=normalizer, training=train_phase, regularizer=regularizer, **kwargs)
                regularization_loss += loss
                relu = tf.nn.relu(norm, name='relu')
                feature_maps = dropout_layer(relu, dropout_rate, train_phase, type=dropout_type)
    return feature_maps, regularization_loss


def conv_upsample(inputs, zoom_factor, filter_size, feature_size, strides=1, dilation_rate=1,
                  initializer=tf.random_normal_initializer(stddev=0.001), regularizer=None,
                  padding='same', trainable=True, interp_method='linear', name_or_scope='conv_upsample'):
    with tf.variable_scope(name_or_scope):
        regularization_loss = 0.
        conv2d = tf.keras.layers.Conv2D(filters=feature_size, kernel_size=filter_size, strides=strides,
                                        padding=padding, dilation_rate=dilation_rate,
                                        kernel_initializer=initializer,
                                        kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                        trainable=trainable, name='conv')
        conv = conv2d(inputs)
        if regularizer == 'l2':
            regularization_loss += tf.reduce_sum(tf.square(conv2d.kernel))
        if regularizer == 'l1':
            regularization_loss += tf.reduce_sum(tf.abs(conv2d.kernel))
        resized_conv = Resize(zoom_factor=zoom_factor, name='resize_conv', interp_method=interp_method)(conv)

    return resized_conv, regularization_loss


def linear_additive_upsample(input_tensor, new_size=2, n_split=4):
    """
    Apply linear additive up-sampling layer, described in paper Wojna et al., The devil is in the decoder,
        https://arxiv.org/abs/1707.05847.

    :param input_tensor: Input tensor.
    :param new_size: The factor of up-sampling.
    :param n_split: The n_split consecutive channels are added together.
    :return: Linearly additively upsampled feature maps.
    """
    with tf.name_scope('linear_additive_upsample'):
        n_channels = input_tensor.get_shape().as_list()[-1]
        input_dim = input_tensor.shape.ndims

        assert n_split > 0 and n_channels % n_split == 0, "Number of feature channels should be divisible by n_split."

        if input_dim == 4:
            upsample = tf.keras.layers.UpSampling2D(size=new_size, name='upsample')(input_tensor)
        elif input_dim == 5:
            upsample = tf.keras.layers.UpSampling3D(size=new_size, name='upsample')(input_tensor)
        else:
            raise TypeError('Incompatible input spatial rank: %d' % input_dim)

        split = tf.split(upsample, n_split, axis=-1)
        split_tensor = tf.stack(split, axis=-1)
        output_tensor = tf.reduce_sum(split_tensor, axis=-1, name='output_tensor')

    return output_tensor


def residual_additive_upsample(inputs, filter_size, strides, feature_size, n_split=2, dropout_rate=0.,
                               initializer=tf.initializers.he_uniform(), normalizer=None,
                               regularizer=None, train_phase=True, trainable=True,
                               name_or_scope='residual_additive_upsample', **kwargs):
    """
    Apply residual linear additive up-sampling layer, described in paper Wojna et al., The devil is in the decoder,
        https://arxiv.org/abs/1707.05847, where the up-sampling are performed with a transposed convolution as well as
        a linear additive up-sampling.

    :param inputs: The input tensor.
    :param filter_size: The kernel size of the transposed convolution.
    :param strides: The strides of the transposed convolution / The factor of up-sampling.
    :param feature_size: The number of filters in the transposed convolution.
    :param n_split: The n_split consecutive channels are added together.
    :param initializer: weight initializer, default as Kaiming uniform initializer
    :param normalizer: type of normalization to use, default is None,
        choose from None, 'batch', 'group', 'layer', 'instance', 'batch_instance'
    :param regularizer: The regularizer for weights.
    :param train_phase: Whether in training or in inference mode.
    :param trainable: Whether add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES.
    :param dropout_rate: Dropout probability.
    :param name_or_scope: The variable scope to open.
    :return: The up-sampled feature maps.
    """
    dropout_type = kwargs.pop('dropout_type', 'regular')
    n_channel = inputs.get_shape().as_list()[-1]
    assert n_channel == feature_size * n_split, "The number of input channels must be the product of output feature " \
                                                "size and the number of splits."
    with tf.variable_scope(name_or_scope):
        regularization_loss = 0.
        deconv2d = tf.keras.layers.Conv2DTranspose(filters=feature_size, kernel_size=filter_size, strides=strides,
                                                   padding='same', use_bias=False, kernel_initializer=initializer,
                                                   kernel_regularizer=regularizer,
                                                   trainable=trainable, name='deconv')
        deconv = deconv2d(inputs)
        if regularizer == 'l2':
            regularization_loss += tf.reduce_sum(tf.square(deconv2d.kernel))
        if regularizer == 'l1':
            regularization_loss += tf.reduce_sum(tf.abs(deconv2d.kernel))
        norm, loss = normalize(deconv, type=normalizer, training=train_phase, regularizer=regularizer, **kwargs)
        regularization_loss += loss
        relu = tf.nn.relu(norm, name='relu')
        dropout = dropout_layer(relu, dropout_rate, train_phase, type=dropout_type)
        upsample = linear_additive_upsample(inputs, strides, n_split)

        return tf.add(dropout, upsample, name='res_upsample'), regularization_loss


# Dropout layer
def dropout_layer(inputs, rate, training, type='regular'):
    """
    Apply regular or spatial dropout to 3D tensors.

    :param inputs: input 3D tensor
    :param rate: dropout rate
    :param training: training mode or inference mode
    :param type: 'regular' or 'spatial'
    :return:
    """
    if rate == 0:
        return inputs
    if type == 'regular':
        outputs = tf.keras.layers.Dropout(rate=rate, name='dropout')(inputs, training=training)
    elif type == 'spatial':
        outputs = tf.keras.layers.SpatialDropout3D(rate=rate, name='dropout')(inputs, training=training)
    else:
        raise NotImplementedError
    return outputs


# Squeeze-and-Excitation Block
def squeeze_excitation_layer(inputs, out_dim, ratio=16, **kwargs):
    """
    Apply Squeeze-and-Excitation layer to 3D tensors.
    Inspired by https://github.com/taki0112/SENet-Tensorflow

    :param inputs: input tensor of shape [n_batch, *vol_shape, channels]
    :param out_dim: output channel size
    :param ratio: reduction ratio, default as 16
    :param kwargs:
    :return:
    """
    with tf.variable_scope(kwargs.pop('scope', 'squeeze_and_excitation')):
        sq = tf.reduce_mean(inputs, axis=(1, 2, 3), name='squeeze')
        fc1 = tf.layers.dense(sq, units=out_dim // ratio, activation=tf.nn.relu, name='fc1')
        fc2 = tf.layers.dense(fc1, units=out_dim, name='fc2')
        ex = tf.nn.sigmoid(fc2, name='excitation')
        ex = tf.reshape(ex, [-1, 1, 1, 1, out_dim])
        scale = inputs * ex
        return scale


# normalization ops
def group_norm(inputs, groups=4, **kwargs):
    """
    Apply group normalization layer to 2D tensors.
    Inspired from https://github.com/shaohua0116/Group-Normalization-Tensorflow/blob/master/ops.py

    :param inputs:
    :param groups: number of groups, set to 1 for layer normalization, set to channel size for instance normalization
    :param eps:
    :return:
    """
    eps = kwargs.pop('eps', 1e-5)
    with tf.variable_scope(kwargs.pop('scope', 'group_norm')):
        x = tf.transpose(inputs, [0, 3, 1, 2])
        N, C, H, W = x.get_shape().as_list()
        G = min(groups, C)
        x = tf.reshape(x, [-1, G, C // G, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4], keepdims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0), dtype=tf.float32)
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.), dtype=tf.float32)
        gamma = tf.reshape(gamma, [1, C, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1])
        output = tf.reshape(x, [-1, C, H, W]) * gamma + beta
        return tf.transpose(output, [0, 2, 3, 1])


def batch_norm(inputs, **kwargs):
    """
    Apply instance normalization layer to 2D tensors.

    :param inputs:
    :param kwargs:
    :return:
    """
    eps = kwargs.pop('eps', 1e-5)
    regularizer = kwargs.pop('regularizer', None)
    with tf.variable_scope(kwargs.pop('scope', 'batch_norm')):
        regularization_loss = 0.
        C = inputs.shape[-1]
        mean, var = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=True)
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0), dtype=tf.float32)
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.), dtype=tf.float32)
        output = tf.nn.batch_normalization(inputs, mean, var, beta, gamma, eps)
        if regularizer == 'l2':
            regularization_loss += tf.reduce_mean(tf.square(gamma))
        if regularizer == 'l1':
            regularization_loss += tf.reduce_mean(tf.abs(gamma))

        return output, regularization_loss


def instance_norm(inputs, **kwargs):
    """
    Apply instance normalization layer to 2D tensors.

    :param inputs:
    :param kwargs:
    :return:
    """
    eps = kwargs.pop('eps', 1e-5)
    regularizer = kwargs.pop('regularizer', None)
    with tf.variable_scope(kwargs.pop('scope', 'instance_norm')):
        regularization_loss = 0.
        C = inputs.shape[-1]
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        x = (inputs - mean) / (tf.sqrt(var + eps))
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0), dtype=tf.float32)
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.), dtype=tf.float32)
        output = x * gamma + beta
        if regularizer == 'l2':
            regularization_loss += tf.reduce_mean(tf.square(gamma))
        if regularizer == 'l1':
            regularization_loss += tf.reduce_mean(tf.abs(gamma))

        return output, regularization_loss


def batch_instance_norm(inputs, **kwargs):
    """
    Apply batch-instance normalization layer to 2D tensors.
    Inspired by https://github.com/taki0112/Batch_Instance_Normalization-Tensorflow

    :param inputs:
    :param kwargs:
    :return:
    """
    eps = kwargs.pop('eps', 1e-5)
    with tf.variable_scope(kwargs.pop('scope', 'batch_instance_norm')):
        C = inputs.shape[-1]
        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2], keep_dims=True)
        x_batch = (inputs - batch_mean) / (tf.sqrt(batch_var + eps))

        ins_mean, ins_var = tf.nn.moments(inputs, axes=[1, 2], keep_dims=True)
        x_ins = (inputs - ins_mean) / (tf.sqrt(ins_var + eps))

        rho = tf.get_variable("rho", [C], initializer=tf.constant_initializer(1.0),
                              constraint=lambda x: tf.clip_by_value(x, clip_value_min=0.0, clip_value_max=1.0))
        gamma = tf.get_variable("gamma", [C], initializer=tf.constant_initializer(1.0), dtype=tf.float32)
        beta = tf.get_variable("beta", [C], initializer=tf.constant_initializer(0.0), dtype=tf.float32)

        x_hat = rho * x_batch + (1 - rho) * x_ins
        output = x_hat * gamma + beta
        return output


def normalize(inputs, type='batch', **kwargs):
    if type is None:
        return inputs
    elif type == 'batch':
        return batch_norm(inputs, **kwargs)
        # training = kwargs.pop('training')
        # return tf.keras.layers.BatchNormalization(name=kwargs.pop('name', 'bn'),
        #                                           **kwargs)(inputs, training=training)
    elif type == 'group':
        return group_norm(inputs, groups=kwargs.pop('groups', 4), **kwargs)
    elif type == 'layer':
        return group_norm(inputs, groups=1, **kwargs)
    elif type == 'instance':
        return instance_norm(inputs, **kwargs)
    elif type == 'batch_instance':
        return batch_instance_norm(inputs, **kwargs)
    else:
        raise NotImplementedError


#######################################################################
# Spatial transformation modules
#######################################################################

class SpatialTransformer(tf.keras.layers.Layer):
    """
    N-D Spatial Transformer Tensorflow / Keras Layer

    The Layer can handle both affine and dense transforms.
    Both transforms are meant to give a 'shift' from the current position.
    Therefore, a dense transform gives displacements (not absolute locations) at each voxel,
    and an affine transform gives the *difference* of the affine matrix from
    the identity matrix.

    If you find this function useful, please cite:
      Unsupervised Learning for Fast Probabilistic Diffeomorphic Registration
      Adrian V. Dalca, Guha Balakrishnan, John Guttag, Mert R. Sabuncu
      MICCAI 2018.

    Originally, this code was based on voxelmorph code, which
    was in turn transformed to be dense with the help of (affine) STN code
    via https://github.com/kevinzakka/spatial-transformer-network

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions

    ToDo:
        The sampling coordinates in this version are defined in the atlas space.
        Need to modify such that the sampling coordinates are defined in the target space.
    """

    def __init__(self,
                 interp_method='linear',
                 indexing='ij',
                 single_transform=False,
                 **kwargs):
        """
        Parameters:
            interp_method: 'linear' or 'nearest'
            single_transform: whether a single transform supplied for the whole batch
            indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian)
                'xy' indexing will have the first two entries of the flow
                (along last axis) flipped compared to 'ij' indexing
        """
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        self.single_transform = single_transform

        assert indexing in ['ij', 'xy'], "indexing has to be 'ij' (matrix) or 'xy' (cartesian)"
        self.indexing = indexing

        super(self.__class__, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input_shape should be a list for two inputs:
        input1: image.
        input2: transform Tensor
            if affine:
                should be a N x N+1 matrix
                *or* a N*(N+1) tensor (which will be reshape to N x (N+1) and an identity row added)
            if not affine:
                should be a *vol_shape x N
        """

        if len(input_shape) > 2:
            raise Exception('Spatial Transformer must be called on a list of length 2.'
                            'First argument is the image, second is the transform.')

        # set up number of dimensions
        self.ndims = len(input_shape[0]) - 2
        self.inshape = input_shape
        vol_shape = input_shape[0][1:-1]
        trf_shape = input_shape[1][1:]

        # the transform is an affine iff:
        # it's a 1D Tensor [dense transforms need to be at least ndims + 1]
        # it's a 2D Tensor and shape == [N+1, N+1].
        #   [dense with N=1, which is the only one that could have a transform shape of 2, would be of size Mx1]
        self.is_affine = len(trf_shape) == 1 or \
                         (len(trf_shape) == 2 and all([f == (self.ndims + 1) for f in trf_shape]))

        # check sizes
        if self.is_affine and len(trf_shape) == 1:
            ex = self.ndims * (self.ndims + 1)
            if trf_shape[0] != ex:
                raise Exception('Expected flattened affine of len %d but got %d'
                                % (ex, trf_shape[0]))

        if not self.is_affine:
            if trf_shape[-1] != self.ndims:
                raise Exception('Offset flow field size expected: %d, found: %d'
                                % (self.ndims, trf_shape[-1]))

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: list with two entries
        """

        # check shapes
        assert len(inputs) == 2, "inputs has to be len 2, found: %d" % len(inputs)
        vol = inputs[0]
        trf = inputs[1]

        # necessary for multi_gpu models...
        vol = tf.reshape(vol, [-1, *self.inshape[0][1:]])
        trf = tf.reshape(trf, [-1, *self.inshape[1][1:]])

        # go from affine
        if self.is_affine:
            trf = tf.map_fn(lambda x: self._single_aff_to_shift(x, vol.shape[1:-1]), trf, dtype=tf.float32)

        # prepare location shift
        if self.indexing == 'xy':  # shift the first two dimensions
            trf_split = tf.split(trf, trf.shape[-1], axis=-1)
            trf_lst = [trf_split[1], trf_split[0], *trf_split[2:]]
            trf = tf.concat(trf_lst, -1)

        # map transform across batch
        if self.single_transform:
            fn = lambda x: self._single_transform([x, trf[0, :]])
            return tf.map_fn(fn, vol, dtype=tf.float32)
        else:
            return tf.map_fn(self._single_transform, [vol, trf], dtype=tf.float32)

    def _single_aff_to_shift(self, trf, volshape):
        if len(trf.shape) == 1:  # go from vector to matrix
            trf = tf.reshape(trf, [self.ndims, self.ndims + 1])

        # note this is unnecessarily extra graph since at every batch entry we have a tf.eye graph
        # trf += tf.eye(self.ndims + 1)[:self.ndims, :]  # add identity, hence affine is a shift from identity
        return affine_to_shift(trf, volshape, shift_center=True)

    def _single_transform(self, inputs):
        return transform(inputs[0], inputs[1], interp_method=self.interp_method)


class Resize(tf.keras.layers.Layer):
    """
    N-D Resize Tensorflow / Keras Layer
    Note: this is not re-shaping an existing volume, but resizing, like scipy's "Zoom"

    If you find this function useful, please cite:
        Anatomical Priors in Convolutional Networks for Unsupervised Biomedical Segmentation,Dalca AV, Guttag J, Sabuncu MR
        CVPR 2018

    Since then, we've re-written the code to be generalized to any
    dimensions, and along the way wrote grid and interpolation functions
    """

    def __init__(self,
                 zoom_factor,
                 interp_method='linear',
                 **kwargs):
        """
        Parameters:
            interp_method: 'linear' or 'nearest'
                'xy' indexing will have the first two entries of the flow
                (along last axis) flipped compared to 'ij' indexing
        """
        self.zoom_factor = zoom_factor
        self.interp_method = interp_method
        self.ndims = None
        self.inshape = None
        super(Resize, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        input_shape should be an element of list of one inputs:
        input1: volume
                should be a *vol_shape x N
        """

        if isinstance(input_shape[0], (list, tuple)) and len(input_shape) > 1:
            raise Exception('Resize must be called on a list of length 1.'
                            'First argument is the image, second is the transform.')

        if isinstance(input_shape[0], (list, tuple)):
            input_shape = input_shape[0]

        # set up number of dimensions
        self.ndims = len(input_shape) - 2
        self.inshape = input_shape

        # confirm built
        self.built = True

    def call(self, inputs):
        """
        Parameters
            inputs: volume of list with one volume
        """

        # check shapes
        if isinstance(inputs, (list, tuple)):
            assert len(inputs) == 1, "inputs has to be len 1. found: %d" % len(inputs)
            vol = inputs[0]
        else:
            vol = inputs

        # necessary for multi_gpu models...
        vol = tf.reshape(vol, [-1, *self.inshape[1:]])

        # map transform across batch
        return tf.map_fn(self._single_resize, vol, dtype=tf.float32)

    def compute_output_shape(self, input_shape):
        output_shape = [input_shape[0]]
        output_shape += [int(f * self.zoom_factor) for f in input_shape[1:-1]]
        output_shape += [input_shape[-1]]
        return tuple(output_shape)

    def _single_resize(self, inputs):
        return resize(inputs, self.zoom_factor, interp_method=self.interp_method)


#######################################################################
# Helper functions
#######################################################################

def b_spline(i, u):
    with tf.name_scope('b_spline'):
        if i == -1:
            return (1 - u) ** 3 / 6
        elif i == 0:
            return (3 * u ** 3 - 6 * u ** 2 + 4) / 6
        elif i == 1:
            return (-3 * u ** 3 + 3 * u ** 2 + 3 * u + 1) / 6
        elif i == 2:
            return u ** 3 / 6


def quaternary(n, rank):
    nums = []
    while n:
        n, r = divmod(n, 4)
        nums.append(r)
    nums += [0] * (rank - len(nums))
    return list(reversed(nums))


#######################################################################
# random affine data augmentation
#######################################################################

def random_affine_matrix_2d(rot_std=np.pi / 12, scl_std=0.1, tra_std=0., she_std=0.1, name='random_affine_params'):
    """
    Generate a random affine transformation matrix.

    :param rot_std: standard deviation of rotation parameters
    :param scl_std: standard deviation of scaling parameters
    :param tra_std: standard deviation of translation parameters
    :param she_std: standard deviation of shearing parameters
    :return: a tensor of shape [1, 12], composed of affine transformation parameters
    """
    a = np.random.normal(0, rot_std, 1)
    sx, sy = np.random.normal(1, scl_std, 2)
    p, q = np.random.normal(0, tra_std, 2)
    hx, hy = np.random.normal(0, she_std, 2)

    # Translation matrix
    Tr = np.asarray([[1, 0, p],
                     [0, 1, q],
                     [0, 0, 1]], dtype=np.float32)

    # Scaling matrix
    Sc = np.asarray([[sx, 0, 0],
                     [0, sy, 0],
                     [0, 0, 1]], dtype=np.float32)

    # Shear matrix
    r = np.random.rand()
    if r < 0.5:
        Sh = np.asarray([[1, hx, 0],
                         [0, 1, 0],
                         [0, 0, 1]], dtype=np.float32)
    else:
        Sh = np.asarray([[1, 0, 0],
                         [hy, 1, 0],
                         [0, 0, 1]], dtype=np.float32)

    # Rotation matrix
    R = np.asarray([[np.cos(a), np.sin(a), 0],
                    [-np.sin(a), np.cos(a), 0],
                    [0, 0, 1]], dtype=np.float32)

    M = np.matmul(Tr,
                  np.matmul(Sc,
                            np.matmul(Sh, R)))

    return tf.reshape(tf.constant(M[:2], dtype=tf.float32), [1, 6], name=name)


def random_affine_augment(inputs, **kwargs):
    """
    Perform data augmentation on inputs tensors.

    :param inputs: a group of tensors ready to be augmented, each of shape [n_batch, *vol_shape, n_channel/n_class] or
        [n_batch, *vol_shape, n_atlas, n_channel/n_class]
    :param kwargs: optional arguments transferred to random_affine_matrix
    """
    name = kwargs.pop('name', 'random_affine_augment')
    affine_augment = kwargs.pop('affine_augment', True)
    interp_methods = kwargs.pop('interp_methods', ['linear'] * len(inputs))
    with tf.name_scope(name):
        affine_params = random_affine_matrix_2d(**kwargs)
        if affine_augment:
            outputs = []
            for i in range(len(inputs)):
                spatial_transform = SpatialTransformer(interp_method=interp_methods[i], single_transform=True)
                if len(inputs[i].get_shape().as_list()) == 6:
                    outputs.append(tf.stack([spatial_transform([inputs[i][..., k, :], affine_params])
                                             for k in range(inputs[i].get_shape().as_list()[-2])], axis=-2))
                else:
                    outputs.append(spatial_transform([inputs[i], affine_params]))
            return outputs
        else:
            return inputs
