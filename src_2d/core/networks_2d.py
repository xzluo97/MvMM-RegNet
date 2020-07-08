# -*- coding: utf-8 -*-
"""
Network architectures for medical image registration.

@author: Xinzhe Luo
"""

from __future__ import print_function, division, absolute_import, unicode_literals
from core.layers_2d import *
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def create_ddf_label_net(target, atlases, dropout_rate,
                         train_phase=True, regularizer=None, normalizer=None,
                         features_root=16, filter_size=3,
                         pool_size=2, num_down_blocks=4, ddf_levels=None,
                         trainable=True, summaries=False, verbose=True,
                         logger=logging, **kwargs):
    """
    Create a network for the prediction of the dense displacement fields between each atlas and the target image with the
    given parametrization.

    :param target: The input target images of shape [n_batch, *vol_shape, n_channel].
    :param atlases: The input probabilistic atlases of shape [n_batch, *vol_shape, n_atlas, n_channel].
    :param dropout_rate: Dropout probability.
    :param train_phase: Whether it is in training or inference mode.
    :param regularizer: Type of regularizer applied to the kernel weights.
    :param normalizer: type of normalization to use, default is None,
        choose from None, 'batch', 'group', 'layer', 'instance', 'batch_instance'
    :param gap_filling: Whether to use gap-filling proposed in:
        J. Fan, X. Cao, P.-T. Yap, and D. Shen, “BIRNet: Brain Image Registration Using Dual-Supervised Fully
        Convolutional Networks,” Med. Image Anal., vol. 54, pp. 193–206, May 2018.
    :param num_filling_blocks:
    :param features_root: The number of feature maps of the first convolution layer.
    :param filter_size: The size of the convolution filter.
    :param pool_size: The size of pooling window of the max pooling layer.
    :param num_down_blocks: The number of downside convolution blocks.
    :param ddf_levels: The levels of network to produce ddf summands.
    :param trainable: Whether add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES
    :param summaries: Flag if summaries should be created.
    :param verbose: If true, print the network architecture settings.
    :param logger: The logging module with specified configuration.
    :returns: output_ddfs - The dense displacement field that register every atlas to the target image, of shape
        [n_batch, *vol_shape, n_atlas, 2]. A dictionary with each key-value pair as ddf of a certain scale.
    """
    ddf_levels = list(range(num_down_blocks + 1)) if ddf_levels is None else list(ddf_levels)
    vol_shape = target.get_shape().as_list()[1:3]
    n_atlas = atlases.get_shape().as_list()[-2]

    gap_filling = kwargs.pop('gap_filling', False)
    dropout_type = kwargs.pop('dropout_type', 'regular')

    # regularization losses
    regularization_loss = []

    if verbose:
        logger.info("Convolutional network for deformable registration with parameterization: "
                    "features root: {features}, filter size: {filter_size}x{filter_size}x{filter_size}, "
                    "pool size: {pool_size}x{pool_size}x{pool_size}, "
                    "number of down-conv blocks: {num_dw_blocks}, "
                    "ddf_levels: {ddf_levels}, "
                    "normalizer: {normalizer}, "
                    "dropout type: {dropout_type}".format(features=features_root, filter_size=filter_size,
                                                          pool_size=pool_size, num_dw_blocks=num_down_blocks,
                                                          ddf_levels=ddf_levels, normalizer=normalizer,
                                                          dropout_type=dropout_type))

    def forward(inputs):
        regularization_loss = 0.
        with tf.variable_scope('encoder'):
            hiddens = OrderedDict()  # Intermediate inputs of each down-sampling layer.
            # down layers
            hiddens[0], loss = conv_block_layer(inputs, num_layers=1, filter_size=7, feature_size=features_root,
                                                regularizer=regularizer, normalizer=normalizer, train_phase=train_phase,
                                                trainable=trainable, name_or_scope='hidden_0')
            regularization_loss += loss

            for layer in range(num_down_blocks):
                dw_h_conv, loss = residual_block_layer(hiddens[layer], filter_size=filter_size,
                                                       feature_size=features_root * 2 ** layer,
                                                       regularizer=regularizer, normalizer=normalizer,
                                                       train_phase=train_phase, trainable=trainable,
                                                       dropout_rate=dropout_rate, dropout_type=dropout_type,
                                                       name_or_scope='down_hidden_layer_%s' % layer)
                regularization_loss += loss
                hiddens[layer + 1], loss = transition_block_layer(dw_h_conv, pool_size=pool_size,
                                                                  filter_size=filter_size, compression_rate=2,
                                                                  regularizer=regularizer, normalizer=normalizer,
                                                                  train_phase=train_phase, trainable=trainable,
                                                                  name_or_scope='transition_down_layer_%s' % layer)
                regularization_loss += loss

        uppers = OrderedDict()  # Intermediate inputs of each up-sampling layer.
        with tf.variable_scope('decoder'):
            # up layers
            uppers[num_down_blocks] = hiddens[num_down_blocks]
            for layer in range(num_down_blocks - 1, -1, -1):
                up_h_conv, loss = residual_additive_upsample(uppers[layer + 1], filter_size=filter_size,
                                                             strides=pool_size, feature_size=features_root * 2 ** layer,
                                                             regularizer=regularizer, normalizer=normalizer,
                                                             train_phase=train_phase, trainable=trainable,
                                                             name_or_scope='additive_upsample_layer_%s' % layer)
                regularization_loss += loss
                # skip-connection whether to use the gap-filling strategy
                if gap_filling:
                    num_filling_blocks = kwargs.pop('num_filling_blocks', (2, 1))
                    skip_features = hiddens[layer]
                    try:
                        gaps = OrderedDict()
                        for k in range(num_filling_blocks[layer]):
                            gaps[(layer, k)], loss = residual_block_layer(skip_features, filter_size=filter_size,
                                                                          feature_size=features_root * 2 ** layer,
                                                                          num_layers=2, regularizer=regularizer,
                                                                          normalizer=normalizer, train_phase=train_phase,
                                                                          trainable=trainable,
                                                                          name_or_scope='gap_layer_%s_block_%s' % (layer, k))
                            regularization_loss += loss
                            skip_features = gaps[(layer, k)]
                    except IndexError:
                        pass

                    skip_connect = tf.add(up_h_conv, skip_features, name='skip_connect')
                else:
                    skip_connect = tf.add(up_h_conv, hiddens[layer], name='skip_connect')

                uppers[layer], loss = residual_block_layer(skip_connect, filter_size=filter_size,
                                                           feature_size=features_root * 2 ** layer,
                                                           regularizer=regularizer, normalizer=normalizer,
                                                           train_phase=train_phase, trainable=trainable,
                                                           dropout_rate=dropout_rate, dropout_type=dropout_type,
                                                           name_or_scope='up_hidden_layer_%s' % layer)
                regularization_loss += loss

        if summaries:
            for k, v in hiddens.items():
                tf.summary.histogram("dw_h_convs_%s" % k, v)

            for k, v in uppers.items():
                tf.summary.histogram("up_h_convs_%s" % k, v)

        return uppers, regularization_loss

    output_ddfs = []
    for i in range(n_atlas):
        with tf.variable_scope('compute_ddfs', reuse=i != 0):
            inputs = tf.concat([target, atlases[..., i, :]], axis=-1)
            uppers, loss = forward(inputs)
            level_ddf = []
            for idx in ddf_levels:
                ddf, l = conv_upsample(uppers[idx], 2 ** idx, filter_size=filter_size,
                                       feature_size=2, regularizer=regularizer,
                                       trainable=trainable, name_or_scope='conv_upsample_ddf_%s' % idx)
                loss += l
                level_ddf.append(ddf)

            regularization_loss.append(loss)
            output_ddfs.append(tf.reduce_sum(tf.stack(level_ddf), axis=0, name='output_ddf_sum'))

    return tf.stack(output_ddfs, axis=-2, name='output_ddfs'), tf.reduce_mean(regularization_loss)

def create_ddf_score_net(target, atlases, dropout_rate,
                         train_phase=True, regularizer=None, normalizer=None,
                         features_root=16, filter_size=3,
                         pool_size=2, num_down_blocks=4, ddf_levels=None,
                         trainable=True, summaries=False, verbose=True,
                         logger=logging, **kwargs):
    """
    Create a network for the prediction of the dense displacement fields between each atlas and the target image with the
    given parametrization.

    :param target: The input target images of shape [n_batch, *vol_shape, n_channel].
    :param atlases: The input probabilistic atlases of shape [n_batch, *vol_shape, n_atlas, n_channel].
    :param dropout_rate: Dropout probability.
    :param train_phase: Whether it is in training or inference mode.
    :param regularizer: Type of regularizer applied to the kernel weights.
    :param normalizer: type of normalization to use, default is None,
        choose from None, 'batch', 'group', 'layer', 'instance', 'batch_instance'
    :param gap_filling: Whether to use gap-filling proposed in:
        J. Fan, X. Cao, P.-T. Yap, and D. Shen, “BIRNet: Brain Image Registration Using Dual-Supervised Fully
        Convolutional Networks,” Med. Image Anal., vol. 54, pp. 193–206, May 2018.
    :param num_filling_blocks:
    :param features_root: The number of feature maps of the first convolution layer.
    :param filter_size: The size of the convolution filter.
    :param pool_size: The size of pooling window of the max pooling layer.
    :param num_down_blocks: The number of downside convolution blocks.
    :param ddf_levels: The levels of network to produce ddf summands.
    :param trainable: Whether add variables to the graph collection GraphKeys.TRAINABLE_VARIABLES
    :param summaries: Flag if summaries should be created.
    :param verbose: If true, print the network architecture settings.
    :param logger: The logging module with specified configuration.
    :returns: output_ddfs - The dense displacement field that register every atlas to the target image, of shape
        [n_batch, *vol_shape, n_atlas, 2]. A dictionary with each key-value pair as ddf of a certain scale
              output_scores - A tensor of shape [n_batch, n_atlas, 1]
    """
    ddf_levels = list(range(num_down_blocks + 1)) if ddf_levels is None else list(ddf_levels)
    vol_shape = target.get_shape().as_list()[1:3]
    n_atlas = atlases.get_shape().as_list()[-2]

    gap_filling = kwargs.pop('gap_filling', False)
    dropout_type = kwargs.pop('dropout_type', 'regular')

    # regularization losses
    regularization_loss = []

    if verbose:
        logger.info("Convolutional network for deformable registration with parameterization: "
                    "features root: {features}, filter size: {filter_size}x{filter_size}x{filter_size}, "
                    "pool size: {pool_size}x{pool_size}x{pool_size}, "
                    "number of down-conv blocks: {num_dw_blocks}, "
                    "ddf_levels: {ddf_levels}, "
                    "normalizer: {normalizer}, "
                    "dropout type: {dropout_type}".format(features=features_root, filter_size=filter_size,
                                                          pool_size=pool_size, num_dw_blocks=num_down_blocks,
                                                          ddf_levels=ddf_levels, normalizer=normalizer,
                                                          dropout_type=dropout_type))

    def forward(inputs):
        regularization_loss = 0.
        with tf.variable_scope('encoder'):
            hiddens = OrderedDict()  # Intermediate inputs of each down-sampling layer.
            # down layers
            hiddens[0], loss = conv_block_layer(inputs, num_layers=1, filter_size=7, feature_size=features_root,
                                                regularizer=regularizer, normalizer=normalizer, train_phase=train_phase,
                                                trainable=trainable, name_or_scope='hidden_0')
            regularization_loss += loss

            for layer in range(num_down_blocks):
                dw_h_conv, loss = residual_block_layer(hiddens[layer], filter_size=filter_size,
                                                       feature_size=features_root * 2 ** layer,
                                                       regularizer=regularizer, normalizer=normalizer,
                                                       train_phase=train_phase, trainable=trainable,
                                                       dropout_rate=dropout_rate, dropout_type=dropout_type,
                                                       name_or_scope='down_hidden_layer_%s' % layer)
                regularization_loss += loss
                hiddens[layer + 1], loss = transition_block_layer(dw_h_conv, pool_size=pool_size,
                                                                  filter_size=filter_size, compression_rate=2,
                                                                  regularizer=regularizer, normalizer=normalizer,
                                                                  train_phase=train_phase, trainable=trainable,
                                                                  name_or_scope='transition_down_layer_%s' % layer)
                regularization_loss += loss

        with tf.variable_scope('fully_connected_layer'):
            x = hiddens[num_down_blocks]  # [n_batch, nx, ny, n_feature]
            x_mean = tf.reduce_mean(x, axis=[1, 2], name='global_pooling')
            x_dense = tf.keras.layers.Dense(units=features_root, activation=tf.nn.leaky_relu, name='dense')(x_mean)
            y = tf.keras.layers.Dense(units=1, name='output')(x_dense)
            score = tf.log(1+tf.exp(y), name='score')

        uppers = OrderedDict()  # Intermediate inputs of each up-sampling layer.
        with tf.variable_scope('decoder'):
            # up layers
            uppers[num_down_blocks] = hiddens[num_down_blocks]
            for layer in range(num_down_blocks - 1, -1, -1):
                up_h_conv, loss = residual_additive_upsample(uppers[layer + 1], filter_size=filter_size,
                                                             strides=pool_size, feature_size=features_root * 2 ** layer,
                                                             regularizer=regularizer, normalizer=normalizer,
                                                             train_phase=train_phase, trainable=trainable,
                                                             name_or_scope='additive_upsample_layer_%s' % layer)
                regularization_loss += loss
                # skip-connection whether to use the gap-filling strategy
                if gap_filling:
                    num_filling_blocks = kwargs.pop('num_filling_blocks', (2, 1))
                    skip_features = hiddens[layer]
                    try:
                        gaps = OrderedDict()
                        for k in range(num_filling_blocks[layer]):
                            gaps[(layer, k)], loss = residual_block_layer(skip_features, filter_size=filter_size,
                                                                          feature_size=features_root * 2 ** layer,
                                                                          num_layers=2, regularizer=regularizer,
                                                                          normalizer=normalizer, train_phase=train_phase,
                                                                          trainable=trainable,
                                                                          name_or_scope='gap_layer_%s_block_%s' % (layer, k))
                            regularization_loss += loss
                            skip_features = gaps[(layer, k)]
                    except IndexError:
                        pass

                    skip_connect = tf.add(up_h_conv, skip_features, name='skip_connect')
                else:
                    skip_connect = tf.add(up_h_conv, hiddens[layer], name='skip_connect')

                uppers[layer], loss = residual_block_layer(skip_connect, filter_size=filter_size,
                                                           feature_size=features_root * 2 ** layer,
                                                           regularizer=regularizer, normalizer=normalizer,
                                                           train_phase=train_phase, trainable=trainable,
                                                           dropout_rate=dropout_rate, dropout_type=dropout_type,
                                                           name_or_scope='up_hidden_layer_%s' % layer)
                regularization_loss += loss

        if summaries:
            for k, v in hiddens.items():
                tf.summary.histogram("dw_h_convs_%s" % k, v)

            for k, v in uppers.items():
                tf.summary.histogram("up_h_convs_%s" % k, v)

        return uppers, regularization_loss, score

    output_ddfs = []
    output_scores = []
    for i in range(n_atlas):
        with tf.variable_scope('compute_ddfs', reuse=i != 0):
            inputs = tf.concat([target, atlases[..., i, :]], axis=-1)
            uppers, loss, score = forward(inputs)
            output_scores.append(score)
            level_ddf = []
            for idx in ddf_levels:
                ddf, l = conv_upsample(uppers[idx], 2 ** idx, filter_size=filter_size,
                                       feature_size=2, regularizer=regularizer,
                                       trainable=trainable, name_or_scope='conv_upsample_ddf_%s' % idx)
                loss += l
                level_ddf.append(ddf)

            regularization_loss.append(loss)
            output_ddfs.append(tf.reduce_sum(tf.stack(level_ddf), axis=0, name='output_ddf_sum'))

    return tf.stack(output_ddfs, axis=-2, name='output_ddfs'), tf.reduce_mean(regularization_loss), \
        tf.stack(output_scores, axis=-2, name='output_scores')
