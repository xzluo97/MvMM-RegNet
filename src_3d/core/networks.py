# -*- coding: utf-8 -*-
"""
Network architectures for medical image registration.

@author: Xinzhe Luo
"""

from __future__ import print_function, division, absolute_import, unicode_literals
from core.layers import *
from collections import OrderedDict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def create_ddf_label_net(target, atlases, dropout_rate, n_atlas,
                         train_phase=True, regularizer=None, normalizer=None,
                         features_root=16, filter_size=3,
                         pool_size=2, num_down_blocks=4, ddf_levels=None,
                         trainable=True, summaries=False, verbose=True,
                         logger=logging, **kwargs):
    """
    Create a network for the prediction of the dense displacement fields between each atlas and the target image with the
    given parametrization.

    :param target: The input target images of shape [n_batch, *vol_shape, n_channel].
    :param atlases: The input probabilistic atlases of shape [n_batch, *vol_shape, n_channel * n_atlas].
    :param dropout_rate: Dropout probability.
    :param train_phase: Whether it is in training or inference mode.
    :param regularizer: Type of regularizer applied to the kernel weights.
    :param normalizer: type of normalization to use, default is None,
        choose from None, 'batch', 'group', 'layer', 'instance', 'batch_instance'
    :param n_atlas: The number of input atlases.
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
        [n_batch, *vol_shape, n_atlas, 3]. A dictionary with each key-value pair as ddf of a certain scale.
    """
    ddf_levels = list(range(num_down_blocks + 1)) if ddf_levels is None else list(ddf_levels)
    vol_shape = target.get_shape().as_list()[1:4]

    dual_encode = kwargs.pop('dual_encode', False)
    gap_filling = kwargs.pop('gap_filling', False)
    separate_ddfs = kwargs.pop('separate_ddfs', False)
    str_network = kwargs.pop('str_network', False)
    dropout_type = kwargs.pop('dropout_type', 'regular')

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

    def encoder(inputs, name_or_scope='encoder'):
        with tf.variable_scope(name_or_scope):
            hiddens = OrderedDict()  # Intermediate inputs of each down-sampling layer.
            # down layers
            hiddens[0] = conv_block_layer(inputs, num_layers=1, filter_size=7, feature_size=features_root,
                                          regularizer=regularizer, normalizer=normalizer,
                                          train_phase=train_phase, trainable=trainable, name_or_scope='hidden_0')

            for layer in range(num_down_blocks):
                dw_h_conv = residual_block_layer(hiddens[layer], filter_size=filter_size,
                                                 feature_size=features_root * 2 ** layer,
                                                 regularizer=regularizer, normalizer=normalizer,
                                                 train_phase=train_phase, trainable=trainable,
                                                 dropout_rate=dropout_rate, dropout_type=dropout_type,
                                                 name_or_scope='down_hidden_layer_%s' % layer)
                hiddens[layer + 1] = transition_block_layer(dw_h_conv, pool_size=pool_size,
                                                            filter_size=filter_size, compression_rate=2,
                                                            regularizer=regularizer, normalizer=normalizer,
                                                            train_phase=train_phase, trainable=trainable,
                                                            name_or_scope='transition_down_layer_%s' % layer)

            return hiddens

    if dual_encode:
        param_share = kwargs.pop('param_share', True)
        if param_share:
            with tf.variable_scope('encoder', reuse=tf.AUTO_REUSE):
                # dual encoder with shared parameters
                target_hiddens = encoder(target)
                atlases_hiddens = encoder(atlases)
        else:
            target_hiddens = encoder(target, name_or_scope='target_encoder')
            atlases_hiddens = encoder(atlases, name_or_scope='atlases_encoder')
        with tf.name_scope('combine_dual_hiddens'):
            hiddens = list(map(lambda x, y: x + y,
                               [target_hiddens[k] for k in range(num_down_blocks + 1)],
                               [atlases_hiddens[k] for k in range(num_down_blocks + 1)]))

    else:
        in_node = tf.concat([target, atlases], -1)
        hiddens = encoder(in_node)

    uppers = OrderedDict()  # Intermediate inputs of each up-sampling layer.
    with tf.variable_scope('decoder'):
        # up layers
        uppers[num_down_blocks] = hiddens[num_down_blocks]
        for layer in range(num_down_blocks - 1, -1, -1):
            up_h_conv = residual_additive_upsample(uppers[layer + 1], filter_size=filter_size,
                                                   strides=pool_size, feature_size=features_root * 2 ** layer,
                                                   regularizer=regularizer, normalizer=normalizer,
                                                   train_phase=train_phase, trainable=trainable,
                                                   name_or_scope='additive_upsample_layer_%s' % layer)
            # skip-connection whether to use the gap-filling strategy
            if gap_filling:
                num_filling_blocks = kwargs.pop('num_filling_blocks', (2, 1))
                skip_features = hiddens[layer]
                try:
                    gaps = OrderedDict()
                    for k in range(num_filling_blocks[layer]):
                        gaps[(layer, k)] = residual_block_layer(skip_features, filter_size=filter_size,
                                                                feature_size=features_root * 2 ** layer, num_layers=2,
                                                                regularizer=regularizer, normalizer=normalizer,
                                                                train_phase=train_phase, trainable=trainable,
                                                                name_or_scope='gap_layer_%s_block_%s' % (layer, k))
                        skip_features = gaps[(layer, k)]
                except IndexError:
                    pass

                skip_connect = tf.add(up_h_conv, skip_features, name='skip_connect')
            else:
                skip_connect = tf.add(up_h_conv, hiddens[layer], name='skip_connect')

            if str_network:
                skip_connect = conv_spatial_transform(skip_connect, filter_size=3,
                                                      name_or_scope='conv_spatial_transform_layer_%s' % layer)

            uppers[layer] = residual_block_layer(skip_connect, filter_size=filter_size,
                                                 feature_size=features_root * 2 ** layer,
                                                 regularizer=regularizer, normalizer=normalizer,
                                                 train_phase=train_phase, trainable=trainable,
                                                 dropout_rate=dropout_rate, dropout_type=dropout_type,
                                                 name_or_scope='up_hidden_layer_%s' % layer)

    # print(output_ddfs.get_shape().as_list())

    if summaries:
        for k, v in hiddens.items():
            tf.summary.histogram("dw_h_convs_%s" % k, v)

        for k, v in uppers.items():
            tf.summary.histogram("up_h_convs_%s" % k, v)

    with tf.variable_scope('compute_ddfs'):
        output_ddfs = [tf.reshape(conv_upsample(uppers[idx], 2 ** idx, filter_size=filter_size,
                                                feature_size=n_atlas * 3, regularizer=regularizer,
                                                trainable=trainable,  name_or_scope='conv_upsample_ddf_%s' % idx),
                                  shape=[-1, ] + vol_shape + [n_atlas, 3], name='ddf_level_%s' % idx)
                       for idx in ddf_levels]

        if separate_ddfs:
            # each level of ddf computed by the sum of previous levels of ddf
            return [tf.reduce_sum(tf.stack(output_ddfs[idx:]), axis=0, name='output_ddf_%s' % idx)
                    for idx in ddf_levels]
            # each level of ddf
            # return output_ddfs
        else:
            return [tf.reduce_sum(tf.stack(output_ddfs), axis=0, name='output_ddf_sum')]


def create_ddf_label_net_v0(target, atlases, dropout_rate, n_atlas,
                            train_phase=True, regularizer=None, features_root=16, filter_size=3, pool_size=2,
                            num_down_blocks=4, ddf_levels=None, trainable=True, summaries=False, verbose=True,
                            logger=logging, **kwargs):
    """
    Create a network for the prediction of the dense displacement fields between each atlas and the target image with the
    given parametrization.

    :param target: The input target images of shape [n_batch, *vol_shape, n_channel].
    :param atlases: The input probabilistic atlases of shape [n_batch, *vol_shape, n_channel * n_atlas].
    :param dropout_rate: Dropout probability.
    :param train_phase: Whether it is in training or inference mode.
    :param regularizer: Type of regularizer applied to the kernel weights.
    :param n_atlas: The number of input atlases.
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
        [n_batch, *vol_shape, n_atlas, 3].
    """
    ddf_levels = list(range(num_down_blocks + 1)) if ddf_levels is None else ddf_levels
    vol_shape = target.get_shape().as_list()[1:4]
    gap_filling = kwargs.pop('gap_filling', False)

    if verbose:
        logger.info("Convolutional network for deformable registration with parameterization: "
                    "features root: {features}, filter size: {filter_size}x{filter_size}x{filter_size}, "
                    "pool size: {pool_size}x{pool_size}x{pool_size}, "
                    "number of down-conv blocks: {num_dw_blocks} "
                    "ddf_levels: {ddf_levels}".format(features=features_root, filter_size=filter_size,
                                                      pool_size=pool_size, num_dw_blocks=num_down_blocks,
                                                      ddf_levels=ddf_levels))

    hiddens = OrderedDict()  # Intermediate inputs of each down-sampling layer.
    uppers = OrderedDict()  # Intermediate inputs of each up-sampling layer.
    in_node = tf.concat([target, atlases], -1)

    # down layers
    hiddens[0] = conv_block_layer(in_node, num_layers=1, filter_size=7, feature_size=features_root,
                                  regularizer=regularizer, train_phase=train_phase, trainable=trainable,
                                  name_or_scope='hidden_0')

    for layer in range(num_down_blocks):
        with tf.variable_scope('downsampling_block_%s' % layer):
            dw_h_conv = residual_block_layer(hiddens[layer], filter_size=filter_size,
                                             feature_size=features_root * 2 ** layer,
                                             regularizer=regularizer, train_phase=train_phase, trainable=trainable,
                                             dropout_rate=dropout_rate)
            hiddens[layer + 1] = transition_block_layer(dw_h_conv, pool_size=pool_size, filter_size=filter_size,
                                                        regularizer=regularizer, train_phase=train_phase,
                                                        trainable=trainable, compression_rate=2)

    # up layers
    uppers[num_down_blocks] = hiddens[num_down_blocks]
    for layer in range(num_down_blocks - 1, -1, -1):
        with tf.variable_scope('upsampling_block_%s' % layer):
            up_h_conv = residual_additive_upsample(uppers[layer + 1], filter_size=filter_size,
                                                   strides=pool_size, feature_size=features_root * 2 ** layer,
                                                   regularizer=regularizer, train_phase=train_phase,
                                                   trainable=trainable)
            # skip-connection whether to use the gap-filling strategy
            if gap_filling:
                num_filling_blocks = kwargs.pop('num_filling_blocks', (2, 1))
                skip_features = hiddens[layer]
                try:
                    gaps = OrderedDict()
                    for k in range(num_filling_blocks[layer]):
                        gaps[(layer, k)] = residual_block_layer(skip_features, filter_size=filter_size,
                                                                feature_size=features_root * 2 ** layer,
                                                                num_layers=2, regularizer=regularizer,
                                                                train_phase=train_phase,
                                                                trainable=trainable,
                                                                name_or_scope='gap%s' % k)
                        skip_features = gaps[(layer, k)]
                except IndexError:
                    pass

                skip_connect = tf.add(up_h_conv, skip_features, name='skip_connect')
            else:
                skip_connect = tf.add(up_h_conv, hiddens[layer], name='skip_connect')

            uppers[layer] = residual_block_layer(skip_connect, filter_size=filter_size,
                                                 feature_size=features_root * 2 ** layer,
                                                 regularizer=regularizer,
                                                 train_phase=train_phase, trainable=trainable,
                                                 dropout_rate=dropout_rate)

    output_ddfs_compact = tf.reduce_sum(tf.stack([conv_upsample(uppers[idx], 2 ** idx, filter_size=filter_size,
                                                                feature_size=n_atlas * 3, regularizer=regularizer,
                                                                train_phase=train_phase, trainable=trainable,
                                                                name_or_scope='conv_upsample_ddf_summand%s' % idx)
                                                  for idx in ddf_levels]), axis=0, name='output_ddfs_compact')

    output_ddfs = tf.reshape(output_ddfs_compact, shape=[-1, ] + vol_shape + [n_atlas, 3], name='output_ddfs')

    # print(output_ddfs.get_shape().as_list())

    if summaries:
        for k, v in hiddens.items():
            tf.summary.histogram("dw_h_convs_%s" % k, v)

        for k, v in uppers.items():
            tf.summary.histogram("up_h_convs_%s" % k, v)

    return [output_ddfs]


def create_unet(target, atlases, dropout_rate,
                train_phase=True, normalizer=None,
                features_root=16, num_down_blocks=4, filter_size=3, pool_size=2,
                # trainable=True, summaries=False, verbose=True, logger=logging, regularizer=None,
                **kwargs):
    """
    A U-Net architecture that predicts pairwise displacement fields.

    :param target:
    :param atlases:
    :param dropout_rate:
    :param featutes_root:
    :param num_down_blocks:
    :param filter_size:
    :param pool_size:
    :param kwargs:
    :return:
    """

    n_atlas = atlases.get_shape().as_list()[-2]
    print(features_root)

    def unet(x_in):
        # down-sampling
        x_enc = [conv_block_layer(x_in, filter_size, features_root, 1, dropout_rate=dropout_rate,
                                  activation=tf.nn.leaky_relu, train_phase=train_phase, normalizer=normalizer,
                                  name_or_scope='input_conv')]
        for i in range(num_down_blocks):
            x_enc.append(conv_block_layer(x_enc[-1], filter_size, features_root * 2 ** (i+1), 1,
                                          strides=pool_size, dropout_rate=dropout_rate, activation=tf.nn.leaky_relu,
                                          train_phase=train_phase, normalizer=normalizer,
                                          name_or_scope='down_conv_%s' % i
                                          )
                         )

        x = x_enc[-1]
        # up-sampling
        for i in range(num_down_blocks - 1, -1, -1):
            # x = deconv_block_layer(x, filter_size, featutes_root * 2 ** i, strides=2, padding='same',
            #                        dropout_rate=dropout_rate, activation=tf.nn.leaky_relu)
            # print(x.get_shape().as_list())
            x = residual_additive_upsample(x, filter_size, features_root * 2 ** i, strides=2,
                                           dropout_rate=dropout_rate, activation=tf.nn.leaky_relu,
                                           train_phase=train_phase, normalizer=normalizer,
                                           name_or_scope='upsample_%s' % i)
            x = tf.concat([x, x_enc[i]], -1)
            x = conv_block_layer(x, filter_size, features_root * 2 ** i, 1,
                                 dropout_rate=dropout_rate, activation=tf.nn.leaky_relu,
                                 train_phase=train_phase, normalizer=normalizer,
                                 name_or_scope='up_conv_%s' % i
                                 )

        x = tf.keras.layers.Conv3D(3, filter_size, padding='same',
                                   kernel_initializer=tf.random_normal_initializer(0, 0.001),
                                   name='output_conv')(x)
        return x

    y = []
    for i in range(n_atlas):
        with tf.variable_scope('unet', reuse=(i != 0)):
            y.append(unet(tf.concat([target, atlases[..., i, :]], -1)))

    return tf.stack(y, axis=-2)

