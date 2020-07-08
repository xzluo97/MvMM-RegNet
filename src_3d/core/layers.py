# -*- coding: utf-8 -*-
"""
Auxiliary functions and operations for network construction, some of which have
been deprecated for high-level modules in TensorFlow.

@author: Xinzhe Luo
"""

from __future__ import print_function, division, absolute_import, unicode_literals
import tensorflow as tf
import numpy as np
from core.utils import transform, resize, affine_to_shift, get_reference_grid_numpy


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
        input_feature_size = inputs.get_shape().as_list()[-1]
        with tf.variable_scope('transition_block_layer'):
            pool = tf.keras.layers.MaxPool3D(pool_size=pool_size, strides=pool_size, name='max_pool')(inputs)
            conv = tf.keras.layers.Conv3D(filters=int(compression_rate * input_feature_size),
                                          kernel_size=filter_size,
                                          padding='same', use_bias=False,
                                          kernel_initializer=initializer,
                                          kernel_regularizer=regularizer, trainable=trainable, name='conv')(pool)
            norm = normalize(conv, type=normalizer, training=train_phase, **kwargs)
            relu = tf.nn.relu(norm, name='relu')
            feature_maps = dropout_layer(relu, dropout_rate, train_phase, type=dropout_type)
    return feature_maps


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
        feature_maps = inputs
        for layer in range(num_layers):
            with tf.variable_scope('res_block_layer%s' % layer):
                conv = tf.keras.layers.Conv3D(filters=feature_size, kernel_size=filter_size, strides=strides,
                                              padding=padding, dilation_rate=dilation_rate, use_bias=False,
                                              kernel_initializer=initializer,
                                              kernel_regularizer=regularizer, trainable=trainable,
                                              name='conv')(feature_maps)
                norm = normalize(conv, type=normalizer, training=train_phase, **kwargs)
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

    return outputs


def conv_block_layer(inputs, filter_size, feature_size, num_layers=2, strides=1, padding='same',
                     dilation_rate=1, dropout_rate=0., initializer=tf.initializers.he_uniform(),
                     activation=tf.nn.relu, normalizer=None, regularizer=None, train_phase=True, trainable=True,
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
        feature_maps = inputs
        for k in range(num_layers):
            with tf.variable_scope('conv_block_layer%d' % k):
                conv = tf.keras.layers.Conv3D(filters=feature_size, kernel_size=filter_size, strides=strides,
                                              padding=padding, dilation_rate=dilation_rate, use_bias=False,
                                              kernel_initializer=initializer,
                                              kernel_regularizer=regularizer, trainable=trainable,
                                              name='conv')(feature_maps)
                norm = normalize(conv, type=normalizer, training=train_phase, **kwargs)
                relu = activation(norm, name='activation')
                feature_maps = dropout_layer(relu, dropout_rate, train_phase, type=dropout_type)
    return feature_maps


def deconv_block_layer(inputs, filter_size, feature_size, strides=1, padding='same',
                       dilation_rate=1, dropout_rate=0., initializer=tf.initializers.he_uniform(),
                       activation=tf.nn.relu, normalizer=None, regularizer=None, train_phase=True, trainable=True,
                       name_or_scope='deconv_block', **kwargs):
    """
    Apply Transposed convolutional block layers to the given inputs.

    :param inputs: Input feature maps.
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
        feature_maps = inputs
        deconv = tf.keras.layers.Conv3DTranspose(filters=feature_size, kernel_size=filter_size, strides=strides,
                                                 padding=padding, dilation_rate=dilation_rate, use_bias=False,
                                                 kernel_initializer=initializer, kernel_regularizer=regularizer,
                                                 trainable=trainable, name='deconv')(feature_maps)
        norm = normalize(deconv, type=normalizer, training=train_phase, **kwargs)
        relu = activation(norm, name='activation')
        feature_maps = dropout_layer(relu, dropout_rate, train_phase, type=dropout_type)
    return feature_maps


def conv_upsample(inputs, zoom_factor, filter_size, feature_size, strides=1, dilation_rate=1,
                  initializer=tf.random_normal_initializer(stddev=0.001), regularizer=None,
                  padding='same', trainable=True, interp_method='linear', name_or_scope='conv_upsample'):
    with tf.variable_scope(name_or_scope):
        conv = tf.keras.layers.Conv3D(filters=feature_size, kernel_size=filter_size, strides=strides,
                                      padding=padding, dilation_rate=dilation_rate,
                                      kernel_initializer=initializer,
                                      kernel_regularizer=regularizer, bias_regularizer=regularizer,
                                      trainable=trainable, name='conv')(inputs)

        resized_conv = Resize(zoom_factor=zoom_factor, name='resize_conv', interp_method=interp_method)(conv)

    return resized_conv


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


def residual_additive_upsample(inputs, filter_size, feature_size, strides, n_split=2, dropout_rate=0.,
                               initializer=tf.initializers.he_uniform(), activation=tf.nn.relu, normalizer=None,
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
        deconv = tf.keras.layers.Conv3DTranspose(filters=feature_size, kernel_size=filter_size, strides=strides,
                                                 padding='same', use_bias=False, kernel_initializer=initializer,
                                                 kernel_regularizer=regularizer,
                                                 trainable=trainable, name='deconv')(inputs)
        norm = normalize(deconv, type=normalizer, training=train_phase, **kwargs)
        relu = activation(norm, name='relu')
        dropout = dropout_layer(relu, dropout_rate, train_phase, type=dropout_type)
        upsample = linear_additive_upsample(inputs, strides, n_split)

        return tf.add(dropout, upsample, name='res_upsample')


def conv_spatial_transform(inputs, filter_size, strides=1, dilation_rate=1, regularizer=None,
                           padding='same', train_phase=True, trainable=True, dropout_rate=0., interp_method='linear',
                           name_or_scope='conv_spatial_transform', **kwargs):
    with tf.variable_scope(name_or_scope):
        ddf = tf.layers.conv3d(inputs, filters=3, kernel_size=filter_size, strides=strides,
                               padding=padding, dilation_rate=dilation_rate,
                               kernel_initializer=tf.random_normal_initializer(stddev=0.01),
                               kernel_regularizer=regularizer, bias_regularizer=regularizer,
                               trainable=trainable, name='ddf')
        warped_features = SpatialTransformer(name='warp_features',
                                             interp_method=interp_method)([inputs, ddf])
        return warped_features


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


# normalization ops
def group_norm(inputs, groups=4, **kwargs):
    """
    Apply group normalization layer to 3D tensors.
    Inspired from https://github.com/shaohua0116/Group-Normalization-Tensorflow/blob/master/ops.py

    :param inputs:
    :param groups: number of groups, set to 1 for layer normalization, set to channel size for instance normalization
    :param eps:
    :return:
    """
    eps = kwargs.pop('eps', 1e-5)
    with tf.variable_scope(kwargs.pop('scope', 'group_norm')):
        x = tf.transpose(inputs, [0, 4, 1, 2, 3])
        N, C, D, H, W = x.get_shape().as_list()
        G = min(groups, C)
        x = tf.reshape(x, [-1, G, C // G, D, H, W])
        mean, var = tf.nn.moments(x, [2, 3, 4, 5], keepdims=True)
        x = (x - mean) / tf.sqrt(var + eps)
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0), dtype=tf.float32)
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.), dtype=tf.float32)
        gamma = tf.reshape(gamma, [1, C, 1, 1, 1])
        beta = tf.reshape(beta, [1, C, 1, 1, 1])
        output = tf.reshape(x, [-1, C, D, H, W]) * gamma + beta
        return tf.transpose(output, [0, 2, 3, 4, 1])


def instance_norm(inputs, **kwargs):
    """
    Apply instance normalization layer to 3D tensors.

    :param inputs:
    :param kwargs:
    :return:
    """
    eps = kwargs.pop('eps', 1e-5)
    with tf.variable_scope(kwargs.pop('scope', 'instance_norm')):
        C = inputs.shape[-1]
        mean, var = tf.nn.moments(inputs, axes=[1, 2, 3], keep_dims=True)
        x = (inputs - mean) / (tf.sqrt(var + eps))
        gamma = tf.get_variable('gamma', [C], initializer=tf.constant_initializer(1.0), dtype=tf.float32)
        beta = tf.get_variable('beta', [C], initializer=tf.constant_initializer(0.), dtype=tf.float32)
        output = x * gamma + beta
        return output


def batch_instance_norm(inputs, **kwargs):
    """
    Apply batch-instance normalization layer to 3D tensors.
    Inspired by https://github.com/taki0112/Batch_Instance_Normalization-Tensorflow

    :param inputs:
    :param kwargs:
    :return:
    """
    eps = kwargs.pop('eps', 1e-5)
    with tf.variable_scope(kwargs.pop('scope', 'batch_instance_norm')):
        C = inputs.shape[-1]
        batch_mean, batch_var = tf.nn.moments(inputs, axes=[0, 1, 2, 3], keep_dims=True)
        x_batch = (inputs - batch_mean) / (tf.sqrt(batch_var + eps))

        ins_mean, ins_var = tf.nn.moments(inputs, axes=[1, 2, 3], keep_dims=True)
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
        training = kwargs.pop('training')
        return tf.keras.layers.BatchNormalization(name=kwargs.pop('name', 'bn'),
                                                  **kwargs)(inputs, training=training)
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


def gaussian_noise_layer(input_layer, std):
    """
    Apply Gaussian noise to the input.

    :param input_layer: Inputs.
    :param std: Standard deviation for the noise.
    :return: Blurred features.
    """
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise


#######################################################################
# Spatial transformation modules
#######################################################################

def warp_grid_ffd(grid, params):
    """
    Warp the mesh grid with free-form deformation.

    :param grid: The reference grids of shape [n_batch, n_1, n_2, ..., n_r, r].
    :param params: The displacement fields of control points, of shape [n_batch, m_1, m_2, ..., m_r, r].
    :return: The warped grids of shape [n_batch, n_1, n_2, ..., n_r, r];
             The dense displacement fields of the reference grids, of shape [n_batch, n_1, ..., n_r, r].
    """
    with tf.name_scope('warp_grid_ffd'):
        n_dim = grid.get_shape().as_list()[-1]  # Get spatial rank r and name it as n_dim.
        n_batch = params.get_shape().as_list()[0]  # Get batch size.
        image_size = tf.constant(grid.get_shape().as_list()[1:-1], tf.float32)
        # Normalize the reference grid in the domain (m_1, m_2, ..., m_r).
        control_points_size = tf.constant(params.get_shape().as_list()[1:-1], tf.float32)
        normalized_coords = grid * tf.reshape(tf.tile(tf.expand_dims(control_points_size, 0), [n_batch, 1]),
                                              [n_batch] + [1] * (len(grid.shape) - 2) + [-1]) / tf.reshape(
            tf.tile(tf.expand_dims(image_size, 0), [n_batch, 1]),
            [n_batch] + [1] * (len(grid.shape) - 2) + [-1])
        # [n_batch, nx, ny, nz, 3]

        # Compute the integer and decimal part of the coordinates
        integer_coords = tf.floor(normalized_coords)  # [n_batch, nx, ny, nz, 3]
        decimal_coords = normalized_coords - integer_coords
        # Compute the weights, of shape [4, n_batch, nx, ny, nz, 3].
        b_spline_weights = tf.stack([b_spline(i, decimal_coords) for i in range(-1, 3)])

        def boundary_replicate(sample_coords0, input_size0):
            """
            Truncate the floor sampling coordinates such that the truncated ones are bounded by the image size.

            :param sample_coords0: The floor sampling coordinates, of shape [n_batch, nx, ny, nz].
            :param input_size0: The image size of the tensor to be re-sampled.
            :return: The bounded floor sampling coordinates, of shape [n_batch, nx, ny, nz].
            """
            return tf.maximum(tf.minimum(sample_coords0, input_size0 - 1), 0)

        # Compute FFD Transformation
        integer_coords = tf.unstack(integer_coords, axis=-1)  # [n_batch, nx, ny, nz]
        sample_coords = [boundary_replicate(tf.cast(x, tf.int32),
                                            tf.cast(control_points_size[i], tf.int32))
                         for i, x in enumerate(integer_coords)]
        sample_coords_minus1 = [boundary_replicate(tf.cast(x - 1., tf.int32),
                                                   tf.cast(control_points_size[i], tf.int32))
                                for i, x in enumerate(integer_coords)]
        sample_coords_plus1 = [boundary_replicate(tf.cast(x + 1., tf.int32),
                                                  tf.cast(control_points_size[i], tf.int32))
                               for i, x in enumerate(integer_coords)]
        sample_coords_plus2 = [boundary_replicate(tf.cast(x + 2., tf.int32),
                                                  tf.cast(control_points_size[i], tf.int32))
                               for i, x in enumerate(integer_coords)]
        sc = (sample_coords_minus1, sample_coords, sample_coords_plus1, sample_coords_plus2)
        quaternary_codes = [quaternary(n, n_dim) for n in range(4 ** n_dim)]

        sz = integer_coords[0].get_shape().as_list()
        batch_coords = tf.tile(tf.reshape(tf.range(sz[0]), [sz[0]] + [1] * (len(sz) - 1)), [1] + sz[1:])

        def make_sample(code):
            return tf.gather_nd(params, tf.stack([batch_coords] + [sc[c][i] for i, c in enumerate(code)], -1))

        samples = tf.stack([make_sample(code) for code in quaternary_codes])  # [64, n_batch, nx, ny, nz, 3]

        weights = tf.stack([tf.reduce_prod(tf.gather(b_spline_weights, code), axis=0)
                            for code in quaternary_codes])  # [64, n_batch, nx, ny, nz, 3]

        ddfs = tf.reduce_sum(weights * samples, axis=0, name='ddfs')

        return tf.add(grid, ddfs, name='warped_grid_ffd'), ddfs


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

def random_affine_matrix(rot_std=np.pi / 12, scl_std=0.1, tra_std=0., she_std=0.1, name='random_affine_params'):
    """
    Generate a random affine transformation matrix.

    :param rot_std: standard deviation of rotation parameters
    :param scl_std: standard deviation of scaling parameters
    :param tra_std: standard deviation of translation parameters
    :param she_std: standard deviation of shearing parameters
    :return: a tensor of shape [1, 12], composed of affine transformation parameters
    """
    ax, ay, az = np.random.normal(0, rot_std, 3)
    sx, sy, sz = np.random.normal(1, scl_std, 3)
    p, q, r = np.random.normal(0, tra_std, 3)
    hxy, hxz, hyx, hyz, hzx, hzy = np.random.normal(0, she_std, 6)

    # Translation matrix
    Tr = np.asarray([[1, 0, 0, p],
                     [0, 1, 0, q],
                     [0, 0, 1, r],
                     [0, 0, 0, 1]], dtype=np.float32)

    # Scaling matrix
    Sc = np.asarray([[sx, 0, 0, 0],
                     [0, sy, 0, 0],
                     [0, 0, sz, 0],
                     [0, 0, 0, 1]], dtype=np.float32)

    # Shear matrix
    Sh = np.asarray([[1, hxy, hxz, 0],
                     [hyx, 1, hyz, 0],
                     [hzx, hzy, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)

    # Rotation matrix about each axis
    Rx = np.asarray([[1, 0, 0, 0],
                     [0, np.cos(ax), -np.sin(ax), 0],
                     [0, np.sin(ax), np.cos(ax), 0],
                     [0, 0, 0, 1]], dtype=np.float32)

    Ry = np.asarray([[np.cos(ay), 0, np.sin(ay), 0],
                     [0, 1, 0, 0],
                     [-np.sin(ay), 0, np.cos(ay), 0],
                     [0, 0, 0, 1]], dtype=np.float32)

    Rz = np.asarray([[np.cos(az), -np.sin(az), 0, 0],
                     [np.sin(az), np.cos(az), 0, 0],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]], dtype=np.float32)

    M = np.matmul(Tr,
                  np.matmul(Sc,
                            np.matmul(Sh,
                                      np.matmul(Rx,
                                                np.matmul(Ry, Rz)))))

    return tf.reshape(tf.constant(M[:3], dtype=tf.float32), [1, 12], name=name)


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
        affine_params = random_affine_matrix(**kwargs)
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
