# -*- coding: utf-8 -*-
"""
Unified Multi-Atlas Segmentation implementations using dense displacement fields for model construction and training.
The optimization is based on the multivariate mixture model of the target image and atlas probabilistic model.

@author: Xinzhe Luo
"""

from __future__ import print_function, division, absolute_import, unicode_literals

import math
import os
import random
import shutil
from torch.utils.data import DataLoader

from core.losses import *
from core.metrics import *
from core.networks import *

tfd = tf.distributions
config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


class UnifiedMultiAtlasSegNet(object):
    """
    A unified multi-atlas segmentation network implementation.

    ToDo:
        transform the atlas label/prob tensors into shape [n_batch, *vol_shape, n_atlas, n_class]
    """

    def __init__(self, input_size: tuple = (64, 64, 64), block_size: tuple = (56, 48, 56), n_blocks=(1, 1, 1),
                 channels: int = 1, n_class: int = 2, n_atlas: int = 5, n_subtypes: tuple = (2, 1,),
                 cost_kwargs=None, aug_kwargs=None, **net_kwargs, ):
        """
        :param input_size: The input size for the network.
        :param block_size: The block size of the input for block reconstruction.
        :param n_blocks: The number of blocks along each axis.
        :param channels: (Optional) number of channels in the input target image.
        :param n_class: (Optional) number of output labels.
        :param n_atlas: The number of atlases within the multivariate mixture model.
        :param n_subtypes: A tuple indicating the number of subtypes within each tissue class, with the first element
            corresponding to the background subtypes.
        :param cost_kwargs: (Optional) kwargs passed to the cost function, e.g. regularizer_type/auxiliary_cost_name.
        :param aug_kwargs: optional data augmentation arguments
        :param net_kwargs: optional network configuration arguments
        """
        # assert n_class == len(n_subtypes), "The length of the subtypes tuple must equal to the number of classes."
        if cost_kwargs is None:
            cost_kwargs = {}
        if aug_kwargs is None:
            aug_kwargs = {}

        # tf.reset_default_graph()
        self.input_size = input_size
        self.n_blocks = n_blocks if isinstance(n_blocks, tuple) else (n_blocks,) * 3
        self.block_size = np.asarray(block_size, dtype=np.int16)
        self.channels = channels
        self.n_class = n_class
        self.n_atlas = n_atlas
        self.n_subtypes = n_subtypes
        self.cost_kwargs = cost_kwargs
        self.aug_kwargs = aug_kwargs
        self.cost_name = cost_kwargs.get('cost_name', 'label_consistency')
        self.net_kwargs = net_kwargs
        self.ddf_levels = list(range(net_kwargs['num_down_blocks'] + 1)) if net_kwargs['ddf_levels'] is None \
            else list(net_kwargs['ddf_levels'])
        self.prob_sigma = cost_kwargs.get('prob_sigma', (1, 2, 4, 8)) \
            if 'multi_scale' in self.cost_name else cost_kwargs.get('prob_sigma', (2,))
        self.prob_eps = cost_kwargs.get('prob_eps', math.exp(-3 ** 2 / 2))
        self.logger = net_kwargs.get("logger", logging)
        self.summaries = net_kwargs.get("summaries", True)
        # initialize regularizer
        self.regularizer_type = self.cost_kwargs.get("regularizer", None)
        self.net_regularizer = None
        self.regularization_coefficient = self.cost_kwargs.get("regularization_coefficient")

        # initialize regularizer for network parameters
        if self.regularizer_type[0]:
            if self.regularizer_type[0] == 'l2':
                self.net_regularizer = tf.keras.regularizers.l2(l=self.regularization_coefficient[0])

            elif self.regularizer_type[0] == 'l1':
                self.net_regularizer = tf.keras.regularizers.l1(l=self.regularization_coefficient[0])
            else:
                raise ValueError("Unknown regularizer for network parameters: %s" % self.regularizer_type[0])

        # define placeholders for inputs
        with tf.name_scope('inputs'):
            # flag for training or inference
            self.train_phase = tf.placeholder(tf.bool, name='train_phase')
            # dropout rate
            self.dropout_rate = tf.placeholder(tf.float32, name='dropout_rate')
            # substructure prior weight
            prior_prob = cost_kwargs.pop("prior_prob", None)
            if prior_prob is None:
                prior_prob =  tf.cast(tf.fill([1, 1, 1, 1, n_class], 1 / self.n_class), dtype=tf.float32)
            self.pi = tf.reshape(prior_prob, shape=[1, 1, 1, 1, n_class], name='prior_prob')
            # input data
            self.data = {'target_image': tf.placeholder(tf.float32,
                                                        [None, input_size[0], input_size[1], input_size[2], channels],
                                                        name='target_image'),
                         'target_label': tf.placeholder(tf.float32,
                                                        [None, input_size[0], input_size[1], input_size[2], n_class],
                                                        name='target_label'),
                         'target_weight': tf.placeholder(tf.float32,
                                                         [None, input_size[0], input_size[1], input_size[2], n_class],
                                                         name='target_weight'),
                         'atlases_image': tf.placeholder(tf.float32,
                                                         [None, input_size[0], input_size[1], input_size[2], n_atlas,
                                                          channels], name='atlases_image'),
                         'atlases_label': tf.placeholder(tf.float32,
                                                         [None, input_size[0], input_size[1], input_size[2], n_atlas,
                                                          n_class], name='atlases_label'),
                         'atlases_weight': tf.placeholder(tf.float32,
                                                          [None, input_size[0], input_size[1], input_size[2], n_atlas,
                                                           n_class], name='atlases_weight')
                         }
            # random affine data augmentation
            self.augmented_data = self._get_augmented_data()

        with tf.variable_scope('network'):
            # compute the dense displacement fields of shape [n_batch, *vol_shape, n_atlas, 3]
            if self.net_kwargs['method'] == 'ddf_label':
                self.ddf = create_ddf_label_net(self.augmented_data['target_image'],
                                                 tf.reshape(self.augmented_data['atlases_image'],
                                                            [-1, input_size[0], input_size[1], input_size[2],
                                                             n_atlas * channels]),
                                                 dropout_rate=self.dropout_rate,
                                                 n_atlas=self.n_atlas,
                                                 train_phase=self.train_phase,
                                                 regularizer=self.net_regularizer,
                                                 **self.net_kwargs)[0]

            elif self.net_kwargs['method'] == 'ddf_label_v0':
                self.ddf = create_ddf_label_net_v0(self.augmented_data['target_image'],
                                                    tf.reshape(self.augmented_data['atlases_image'],
                                                               [-1, input_size[0], input_size[1], input_size[2],
                                                                n_atlas * channels]),
                                                    dropout_rate=self.dropout_rate,
                                                    n_atlas=self.n_atlas,
                                                    train_phase=self.train_phase,
                                                    regularizer=self.net_regularizer,
                                                    **self.net_kwargs)

            elif self.net_kwargs['method'] == 'unet':
                self.ddf = create_unet(self.augmented_data['target_image'],
                                       self.augmented_data['atlases_image'],
                                       dropout_rate=self.dropout_rate,
                                       train_phase=self.train_phase,
                                       regularizer=self.net_regularizer,
                                       **self.net_kwargs)
            else:
                raise ValueError("Unknown method: %s" % self.net_kwargs['method'])

        # integrate velocity fields by scaling and squaring
        if self.net_kwargs['diffeomorphism']:
            self.int_steps = self.net_kwargs.pop('int_steps', 8)
            self.vec = self.ddf / (2**self.int_steps)
            self.ddf = utils.integrate_vec(self.vec, self.int_steps)

        with tf.variable_scope('loss'):
            # get target probability map, sigma is set to self.prob_sigma[0] (the finest scale) as default
            self.target_prob = utils.get_prob_from_label(self.augmented_data['target_label'],
                                                            sigma=self.prob_sigma[0],
                                                            eps=self.prob_eps)

            # get warped atlases probs/labels from each scale of ddf, each of shape [n_batch, *vol_shape, n_atlas, n_class]
            self.warped_atlases_prob = self._get_warped_atlases_prob(self.augmented_data['atlases_label'],
                                                                     self.ddf, interp_method='linear')

            # get warped atlases joint probability map of each scale, of shape [n_batch, *vol_shape, n_class]
            self.atlases_joint_prob = [utils.get_joint_prob(prob) for prob in self.warped_atlases_prob]

            # get loss function and joint distributions
            self.cost = self._get_cost(self.regularizer_type)
            self.pretrain_cost = self._get_pretrain_cost()

        # get segmentation
        self.segmenter = utils.get_segmentation(self.atlases_joint_prob[0])

        # get joint posterior distribution over target images
        # self.joint_post_probs = utils.get_joint_post_probs(self._joint_probs)

        # get variables and update-ops
        self.trainable_variables = tf.trainable_variables(scope='network')
        self.training_variables = tf.global_variables(scope='network')
        self.update_ops = tf.compat.v1.get_collection(tf.GraphKeys.UPDATE_OPS, scope='network')

        # set global step and moving average
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        # self.ema = tf.train.ExponentialMovingAverage(decay=0.9999, num_updates=self.global_step)
        self.variables_to_restore = self.training_variables

        # get gradients
        self.gradients_node = tf.gradients(self.cost, self.trainable_variables, name='gradients')

        with tf.name_scope('metrics'):
            self.average_dice = OverlapMetrics(n_class).averaged_foreground_dice(self.augmented_data['target_label'],
                                                                                 self.segmenter)
            self.myocardial_dice = OverlapMetrics(n_class).class_specific_dice(self.augmented_data['target_label'],
                                                                               self.segmenter, i=1)
            self.jaccard = OverlapMetrics(n_class).averaged_foreground_jaccard(self.augmented_data['target_label'],
                                                                               self.segmenter)
            self.ddf_norm = tf.norm(self.ddf, name='ddf_norm')

    def _get_augmented_data(self, type=''):
        """
        Data augmentation using affine transformations.
        :param type: type of augmentation
        :return: The augmented data in training stage, whereas the original data in validation/test stage.
        """
        with tf.name_scope('augment_data'):
            def true_fn():
                augmented_data = dict(zip(['target_image', 'target_label', 'target_weight'],
                                          random_affine_augment([self.data['target_image'], self.data['target_label'],
                                                                 self.data['target_weight']],
                                                                interp_methods=['linear', 'nearest', 'linear'],
                                                                **self.aug_kwargs)))
                # augmented_data.update(dict(zip(['atlases_image', 'atlases_label'], random_affine_augment([
                # self.data['atlases_image'], self.data['atlases_label']], interp_methods=['linear', 'nearest'],
                # **self.aug_kwargs))))
                augmented_data.update(dict(zip(['atlases_image', 'atlases_label', 'atlases_weight'],
                                               [self.data['atlases_image'], self.data['atlases_label'],
                                                self.data['atlases_weight']])))
                return augmented_data

            return tf.cond(self.train_phase, true_fn, lambda: self.data)

    def _get_pretrain_cost(self):
        with tf.name_scope('pretrain_cost'):
            return tf.reduce_mean(self.ddf ** 2, name='pretrain_cost')

    def _get_cost(self, regularizer_type=None):
        """
        Constructs the cost function, Optional arguments are:
        regularization_coefficient: weight of the regularization term

        :param regularizer_type: type of regularization

        :return: loss - The weighted sum of the negative log-likelihood and the regularization term, as well as the
            auxiliary guidance term if designated;
                 joint_probs - The joint distribution of images and tissue classes, of shape [n_batch, *vol_shape,
            n_class].
        """

        with tf.name_scope('cost_function'):
            if self.cost_name == 'label_consistency':
                loss = LabelConsistencyLoss(**self.cost_kwargs).loss(self.target_prob,
                                                                     self.atlases_joint_prob[0],
                                                                     self.pi)

            elif self.cost_name == 'multi_scale_label_consistency':
                loss = LabelConsistencyLoss(**self.cost_kwargs).multi_scale_loss(self.target_prob,
                                                                                 self.atlases_joint_prob,
                                                                                 self.pi)

            elif self.cost_name == 'dice':
                # Dice loss between two probabilistic labels
                loss = DiceLoss().loss(self.target_prob, self.atlases_joint_prob[0])

            elif self.cost_name == 'multi_scale_dice':
                loss = DiceLoss().multi_scale_loss(self.target_prob, self.atlases_joint_prob)

            elif self.cost_name == 'cross_entropy':
                # class conditional probabilities over all atlases, of shape [n_batch, *vol_shape, n_class]
                loss = tf.reduce_mean(
                    tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.augmented_data['target_label'],
                                                               logits=self.atlases_joint_prob[0]),
                    name='cross_entropy')

            elif self.cost_name == 'SSD':
                loss = tf.reduce_mean(tf.square(self.target_prob - self.atlases_joint_prob[0]), name='SSD')

            elif self.cost_name == 'LNCC':
                warped_atlases_image = self._get_warped_atlases(self.augmented_data['atlases_image'], self.ddf)
                loss = CrossCorrelation().loss(self.augmented_data['target_image'],
                                               tf.squeeze(warped_atlases_image, axis=-2))

            elif self.cost_name == 'L2_norm':
                loss = tf.reduce_mean(tf.square(self.ddf))

            elif self.cost_name == 'mvmm_net_gmm':
                warped_atlases_weight = self._get_warped_atlases(self.augmented_data['atlases_weight'], self.ddf)
                loss = MvMMNetLoss(**self.cost_kwargs).loss_weight([tf.expand_dims(self.target_prob, -2),
                                                                    self.warped_atlases_prob[0]],
                                                                   [tf.expand_dims(self.augmented_data['target_weight'],
                                                                                   -2), warped_atlases_weight],
                                                                   self.pi)

            elif self.cost_name == 'mvmm_net_ncc':
                warped_atlases_weight = self._get_warped_atlases(self.augmented_data['atlases_weight'], self.ddf)
                loss = MvMMNetLoss(**self.cost_kwargs).loss_weight([tf.expand_dims(self.target_prob, -2),
                                                                    self.warped_atlases_prob[0]],
                                                                   [tf.expand_dims(self.augmented_data['target_weight'],
                                                                                   -2), warped_atlases_weight],
                                                                   self.pi)

            elif self.cost_name == 'mvmm_net_lecc':
                warped_atlases_weight = self._get_warped_atlases(self.augmented_data['atlases_weight'], self.ddf)
                loss = MvMMNetLoss(**self.cost_kwargs).loss_weight([tf.expand_dims(self.target_prob, -2),
                                                                    self.warped_atlases_prob[0]],
                                                                   [tf.expand_dims(self.augmented_data['target_weight'],
                                                                                   -2), warped_atlases_weight],
                                                                   self.pi)

            elif self.cost_name == 'mvmm_net_mask':
                loss = MvMMNetLoss(**self.cost_kwargs).loss_mask([tf.expand_dims(self.target_prob, -2),
                                                                  self.warped_atlases_prob[0]], self.pi)
            else:
                raise NotImplementedError

            # add regularization loss
            if regularizer_type[0] in ('l2', 'l1') and self.regularization_coefficient[0]:
                regularization_loss = tf.keras.regularizers.get(identifier=regularizer_type[0])
                loss += regularization_loss

            return loss

    def _get_warped_atlases_prob(self, atlases_label, ddf, interp_method='linear', **kwargs):
        """
        Warp multiple atlases with the dense displacement fields as the network outputs and produce warped atlases
        probability maps.
        """
        with tf.name_scope('warp_atlases_prob'):
            spatial_transform = SpatialTransformer(interp_method, name='warp_atlases_prob')
            eps = kwargs.pop("eps", self.prob_eps)
            warped_atlases_probs = []
            for idx in range(len(self.prob_sigma)):
                warped_atlases_probs.append(
                    tf.stack([spatial_transform([utils.get_prob_from_label(atlases_label[..., n, :],
                                                                              self.prob_sigma[idx],
                                                                              eps=eps),
                                                 ddf[..., n, :]]) for n in range(self.n_atlas)], axis=-2))

            return warped_atlases_probs

    def _get_warped_atlases(self, atlases, ddf, **kwargs):
        with tf.name_scope('warp_atlases'):
            spatial_transform = SpatialTransformer(name='warp_atlases', **kwargs)
            warped_atlases = tf.stack([spatial_transform([atlases[..., i, :], ddf[..., i, :]])
                                             for i in range(self.n_atlas)], axis=-2)
            return warped_atlases

    def save(self, saver, sess, model_path, **kwargs):
        """
        Saves the current session to a checkpoint

        :param saver: the TensorFlow saver
        :param sess: current session
        :param model_path: path to file system location
        :param latest_filename: Optional name for the protocol buffer file that will contains the list of most recent
        checkpoints.
        """

        save_path = saver.save(sess, model_path, **kwargs)
        self.logger.info("Model saved to file: %s" % save_path)
        return save_path

    def restore(self, sess, model_path, **kwargs):
        """
        Restores a session from a checkpoint

        :param sess: current session instance
        :param model_path: path to file system checkpoint location
        """

        saver = tf.train.Saver(**kwargs)
        saver.restore(sess, model_path)
        self.logger.info("Model restored from file: %s" % model_path)

    def __str__(self):
        # Todo: to make the print more complete and pretty
        return "\n################ Network Parameter Settings ################\n" \
               "input_size= {}, num_blocks= {}, num_channels= {}, num_classes= {}, num_atlases= {}, " \
               "num_subtypes= {}, \n" \
               "ddf_levels= {}, features_root= {}, dropout_rate= {}, \n" \
               "cost_name= {}, prob_sigma= {}, regularizer_type= {}, " \
               "regularizer_coefficient= {}".format(self.input_size, self.n_blocks, self.channels,
                                                    self.n_class, self.n_atlas, self.n_subtypes,
                                                    self.net_kwargs.get("ddf_levels"),
                                                    self.net_kwargs.get("features_root"),
                                                    self.net_kwargs.get("dropout_rate"),
                                                    self.cost_name, self.prob_sigma,
                                                    self.regularizer_type,
                                                    self.regularization_coefficient)

    __repr__ = __str__


class NetForPrediction(UnifiedMultiAtlasSegNet):
    """
    Model prediction for the unified multi-atlas segmentation network.
    """

    def __init__(self, input_scale=1, test_input_size=(64, 64, 64), input_size=(64, 64, 64), block_size=(56, 48, 56),
                 n_blocks=(1, 1, 1), channels=1, n_class=2, n_atlas=1, n_subtypes=(2, 1,), cost_kwargs=None,
                 test_n_class=8, **net_kwargs):
        """
        :param input_scale: The input scale for the network.
        :param test_input_size: The test input size.
        :param input_size: The input size for the network.
        :param block_size: The block size of the input for block reconstruction.
        :param n_blocks: The number of blocks along each axis.
        :param channels: (Optional) number of channels in the input target image.
        :param n_class: (Optional) number of output labels.
        :param n_atlas: The number of atlases within the multivariate mixture model.
        :param n_subtypes: A tuple indicating the number of subtypes within each tissue class, with the first element
            corresponding to the background subtypes.
        :param cost_kwargs: (Optional) kwargs passed to the cost function, e.g. regularizer_type/auxiliary_cost_name.
        """

        super(NetForPrediction, self).__init__(input_size, block_size, n_blocks, channels, n_class, n_atlas, n_subtypes,
                                               cost_kwargs, **net_kwargs)
        self.input_scale = input_scale
        self.test_input_size = test_input_size
        self.test_n_class = test_n_class
        # resize the dense displacement fields
        self.resized_ddf = utils.pad_to_shape_image(Resize(zoom_factor=2 ** input_scale,
                                                              name='resize_ddf')(self.ddf[..., 0, :]),
                                                       shape=test_input_size, mode='tf')
        # placeholders for original atlases
        self.test_atlases_label = tf.placeholder(tf.float32, [1, test_input_size[0], test_input_size[1],
                                                              test_input_size[2], n_atlas, test_n_class],
                                                 name='test_atlases_label')
        self.test_atlases_image = tf.placeholder(tf.float32, [1, test_input_size[0], test_input_size[1],
                                                              test_input_size[2], n_atlas, self.channels],
                                                 name='test_atlases_image')
        self.test_atlases_weight = tf.placeholder(tf.float32, [1, test_input_size[0], test_input_size[1],
                                                               test_input_size[2], n_atlas, test_n_class],
                                                  name='test_atlases_weight')

        # warp original atlases using resized ddf
        self.resized_warped_atlas_image = SpatialTransformer(interp_method='linear',
                                                             name='warp_resized_atlas_image')(
            [self.test_atlases_image[..., 0, :],
             utils.crop_to_shape(
                 self.resized_ddf,
                 self.test_input_size,
                 mode='tf')])

        self.resized_warped_atlas_label = SpatialTransformer(interp_method='nearest',
                                                             name='warp_resized_atlas_label')(
            [self.test_atlases_label[..., 0, :],
             utils.crop_to_shape(
                 self.resized_ddf,
                 self.test_input_size,
                 mode='tf')])

        self.resized_warped_atlas_weight = SpatialTransformer(interp_method='linear',
                                                              name='warp_resized_atlas_weight')(
            [self.test_atlases_weight[..., 0, :],
             utils.crop_to_shape(
                 self.resized_ddf,
                 self.test_input_size,
                 mode='tf')])

    def predict_scale(self, sess, model_data, test_data, dropout_rate):
        """
        Restore the model to make inference for the test data with resized dense displacement fields.

        :param sess: The session for predictions.
        :param model_data: The input data for network prediction.
        :param test_data: The test data for model inference.
        :param dropout_rate: dropout probability for network inference;
        :return: image_pred - The predicted array representing the warped atlas images;
                 label_pred - The predicted predictions representing the warped atlas labels;
                 metrics - A dictionary combining list of various evaluation metrics.
        """

        self.logger.info("Start predicting warped test atlas!")

        image_pred, label_pred, weight_pred, \
        ddf, num_neg_jacob = sess.run((self.resized_warped_atlas_image,
                                        self.resized_warped_atlas_label,
                                        self.resized_warped_atlas_weight,
                                        self.ddf, self.num_neg_jacob),
                                       feed_dict={self.data['target_image']: model_data['target_image'],
                                                  self.data['atlases_image']: model_data['atlases_image'],
                                                  # self.data['atlases_label']: model_data['atlases_label'],
                                                  # self.pi: test_pi[i],
                                                  self.test_atlases_image: test_data['atlases_image'],
                                                  self.test_atlases_label: test_data['atlases_label'],
                                                  self.test_atlases_weight: test_data['atlases_weight'],
                                                  self.dropout_rate: dropout_rate,
                                                  self.train_phase: False})

        Overlap = OverlapMetrics(n_class=self.test_n_class, mode='np')
        dice = Overlap.averaged_foreground_dice(y_true=test_data['target_label'], y_seg=label_pred)
        class_specific_dice = [Overlap.class_specific_dice(y_true=test_data['target_label'], y_seg=label_pred, i=k)
                               for k in range(self.test_n_class)]

        # self.logger.info("Metrics before registration: Dice= {:.4f}, "
        #                  "Myocardial Dice= {:.4f}".format(
        #     Overlap.averaged_foreground_dice(y_true=test_data['target_label'],
        #                                      y_seg=test_data['atlases_label'].squeeze(-2)),
        #     Overlap.class_specific_dice(y_true=test_data['target_label'],
        #                                 y_seg=test_data['atlases_label'].squeeze(-2), i=1)
        # )
        # )

        metrics = {'Dice': dice,
                   'Jaccard': Overlap.averaged_foreground_jaccard(y_true=test_data["target_label"],
                                                                  y_seg=label_pred),
                   'Myocardial Dice': class_specific_dice[1],
                   'LA Dice': class_specific_dice[2],
                   'LV Dice': class_specific_dice[3],
                   'RA Dice': class_specific_dice[4],
                   'RV Dice': class_specific_dice[5],
                   'AO Dice': class_specific_dice[6],
                   'PA Dice': class_specific_dice[7],
                   '# Negative Jacobians': num_neg_jacob}

        self.logger.info("Metrics after registration: "
                         "Dice= {:.4f}, Myocardial Dice= {:.4f}, "
                         "LA Dice= {:.4f}, LV Dice= {:.4f}, "
                         "RA Dice= {:.4f}, RV Dice= {:.4f}, "
                         "AO Dice= {:.4f}, PA Dice= {:.4f}, "
                         "# Negative Jacobians= {:.1f}".format(metrics['Dice'], metrics['Myocardial Dice'],
                                                               metrics['LA Dice'], metrics['LV Dice'],
                                                               metrics['RA Dice'], metrics['RV Dice'],
                                                               metrics['AO Dice'], metrics['PA Dice'],
                                                               metrics['# Negative Jacobians']))

        return image_pred, label_pred, weight_pred, ddf, metrics


class Trainer(object):
    """
    Trains a unified multi-atlas segmentation network.
    """

    def __init__(self, net, batch_size=1, norm_grads=False, optimizer_name="momentum", learning_rate=0.001,
                 num_workers=0, opt_kwargs=None):
        """
        :param net: The network instance to train.
        :param batch_size: The size of training batch.
        :param norm_grads: (Optional) true if normalized gradients should be added to the summaries.
        :param optimizer_name: (Optional) name of the optimizer to use (momentum or adam).
        :param learning_rate: learning rate
        :param num_workers: How many sub-processes to use for data loading.
            0 means that the data will be loaded in the main process. (default: 0)
        :param opt_kwargs: (Optional) kwargs passed to the learning rate (momentum opt) and to the optimizer.
        """
        if opt_kwargs is None:
            opt_kwargs = {}
        self.net = net
        self.batch_size = batch_size
        self.norm_grads = norm_grads
        self.optimizer_name = optimizer_name
        self.num_workers = num_workers
        self.opt_kwargs = opt_kwargs
        self.learning_rate = learning_rate

    def _get_optimizer(self, cost, global_step, clip_gradient=False, **kwargs):
        optimizer_name = kwargs.pop('optimizer', self.optimizer_name)
        decay_steps = kwargs.pop('decay_step', 100000)
        trainable_variables = self.net.trainable_variables

        # variables_to_average = trainable_variables + tf.moving_average_variables()
        # ema = self.net.ema
        self.decay_rate = self.opt_kwargs.get("decay_rate", 0.999)
        if optimizer_name == "momentum":
            init_lr = kwargs.pop("lr", 0.2)
            momentum = self.opt_kwargs.get("momentum", 0.9)
            self.net.logger.info("SGD optimizer with initial lr: {:.2e}, momentum: {:.2f}, "
                                 "decay steps: {:d}, decay rate: {:.2f}".format(init_lr, momentum, decay_steps,
                                                                                self.decay_rate))
            learning_rate_node = tf.train.exponential_decay(learning_rate=init_lr,
                                                            global_step=global_step,
                                                            decay_steps=decay_steps,
                                                            decay_rate=self.decay_rate,
                                                            staircase=True, name='learning_rate')
            optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate_node, momentum=momentum,
                                                   **self.opt_kwargs)

        elif optimizer_name == "sgd":
            init_lr = kwargs.pop('lr', 0.1)
            self.net.logger.info("SGD optimizer with initial lr: {:.2e}, "
                                 "decay steps: {:d}, decay rate: {:.2f}".format(init_lr, decay_steps,
                                                                                self.decay_rate))
            learning_rate_node = tf.train.exponential_decay(learning_rate=init_lr,
                                                            global_step=global_step,
                                                            decay_steps=decay_steps,
                                                            decay_rate=self.decay_rate,
                                                            staircase=True, name='learning_rate')
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_node, **self.opt_kwargs)

        elif optimizer_name == 'rmsprop':
            init_lr = kwargs.pop('lr', 0.001)
            self.net.logger.info("RMSprop optimizer with initial lr: {:.2e}".format(init_lr))
            learning_rate_node = tf.Variable(init_lr, trainable=False, name='learning_rate')
            optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate_node, **self.opt_kwargs)

        elif optimizer_name == "adam":
            init_lr = kwargs.pop('lr', 0.001)
            self.net.logger.info("Adam optimizer with initial lr: {:.2e}".format(init_lr))
            learning_rate_node = tf.Variable(init_lr, trainable=False, name='learning_rate')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_node, **self.opt_kwargs)

        elif optimizer_name == 'adam-clr':
            from core.clr import cyclic_learning_rate
            init_lr = kwargs.pop('lr', 0.001)
            step_size = kwargs.pop("step_size", 100000)
            gamma = kwargs.pop("gamma", 0.99999)
            self.net.logger.info("Adam optimizer with cyclic learning rate, initial lr: {:.2e} "
                                 "step_size: {:d}, gamma: {:.5e}".format(init_lr, step_size, gamma))
            learning_rate_node = cyclic_learning_rate(global_step, learning_rate=init_lr,
                                                      max_lr=init_lr * 10,
                                                      step_size=step_size, gamma=gamma,
                                                      mode='exp_range')
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_node, **self.opt_kwargs)

        elif optimizer_name == 'radam':
            from core.radam import RAdamOptimizer
            init_lr = kwargs.pop('lr', 0.001)
            total_steps = kwargs.pop('total_steps', 0)
            min_lr = kwargs.pop('min_lr', 1e-5)
            self.net.logger.info("Rectified Adam optimizer with initial lr: {:.2e}, minimum lr: {:.2e} "
                                 "total steps: {:d}".format(init_lr, min_lr, total_steps))
            learning_rate_node = tf.Variable(init_lr, trainable=False, name='learning_rate')
            optimizer = RAdamOptimizer(learning_rate=learning_rate_node, total_steps=total_steps,
                                       min_lr=min_lr, **self.opt_kwargs)

        else:
            raise ValueError("Unknown optimizer: %s" % optimizer_name)

        if clip_gradient:
            gradients, variables = zip(*optimizer.compute_gradients(cost, var_list=trainable_variables))

            # clip by global norm
            capped_grads, _ = tf.clip_by_global_norm(gradients, 1.0)
            '''
            # clip by individual norm
            capped_grads = [None if grad is None else tf.clip_by_norm(grad, 1.0) for grad in gradients]
            '''
            opt_op = optimizer.apply_gradients(zip(capped_grads, variables), global_step=global_step)
        else:
            opt_op = optimizer.minimize(cost, global_step=global_step, var_list=trainable_variables)

        update_op = kwargs.pop('update_op', None)
        if update_op:
            train_op = tf.group([opt_op, self.net.update_ops, update_op])
        else:
            train_op = tf.group([opt_op, self.net.update_ops])

        return optimizer, train_op, init_lr, learning_rate_node

    def _initialize(self, training_iters, self_iters, decay_epochs, epochs, clip_gradient, restore_model_path,
                    save_model_path, restore, prediction_path, pretrain_epochs):
        self.training_iters = training_iters
        self.self_iters = self_iters
        self.decay_epochs = decay_epochs
        self.epochs = epochs
        opt_decay_steps = training_iters * self_iters * decay_epochs

        if self.net.summaries and self.norm_grads:
            self.norm_gradients_node = tf.Variable(tf.constant(0.0, shape=[len(self.net.gradients_node)]),
                                                   name='norm_gradients')
            tf.summary.histogram('norm_grads', self.norm_gradients_node)

        # create summary protocol buffers for training metrics
        with tf.name_scope('Training_metrics_summaries'):
            tf.summary.scalar('Training_Loss', tf.reduce_mean(self.net.cost))
            #     tf.summary.scalar('Training_Accuracy', tf.reduce_mean(self.net.acc))
            #     tf.summary.scalar('Training_AUC', tf.reduce_mean(self.net.auc))
            #     tf.summary.scalar('Training_Sensitivity', tf.reduce_mean(self.net.sens))
            #     tf.summary.scalar('Training_Specificity', tf.reduce_mean(self.net.spec))
            tf.summary.scalar('Training_Average_Dice', self.net.average_dice)
            tf.summary.scalar('Training_Myocardial_Dice', self.net.myocardial_dice)
            tf.summary.scalar('Training_Jaccard', self.net.jaccard)
            tf.summary.scalar('Training_DDF_Norm', self.net.ddf_norm)

        # add regularization terms
        regularization_operand = self.net.cost_kwargs.pop('regularization_operand', 'displacement')
        if regularization_operand == 'displacement':
            operand = self.net.ddf
        elif regularization_operand == 'vector':
            assert self.net.net_kwargs['diffeomorphism'], 'Must be diffeomorphic registration if the ' \
                                                          'regularization operand is the vector field!'
            operand = self.net.vec
        else:
            raise NotImplementedError

        # add membrane energy
        if self.net.regularizer_type[1] == 'membrane_energy':
            with tf.name_scope('membrane_energy'):
                MembraneEnergy = LocalDisplacementEnergy(energy_type='membrane')
                n_subject = operand.get_shape().as_list()[-2]
                membrane_energy = tf.reduce_sum([
                    MembraneEnergy.compute_displacement_energy(operand[..., i, :],
                                                               self.net.regularization_coefficient[1])
                    for i in range(n_subject)])
                if regularization_operand == 'vector':
                    membrane_energy *= 2 ** self.net.int_steps
                setattr(self.net, 'membrane_energy', membrane_energy)
                self.net.cost += membrane_energy

        # add bending energy
        if self.net.regularizer_type[2] == 'bending_energy':
            with tf.name_scope('bending_energy'):
                bending_energy_increment_rate = self.net.cost_kwargs.pop('bending_energy_increment_rate')
                bending_energy_weight = tf.train.exponential_decay(self.net.regularization_coefficient[2],
                                                                   global_step=self.net.global_step,
                                                                   decay_steps=training_iters * self_iters,
                                                                   decay_rate=bending_energy_increment_rate,
                                                                   staircase=True, name='bending_energy_weight')
                BendingEnergy = LocalDisplacementEnergy(energy_type='bending')
                n_subject = operand.get_shape().as_list()[-2]
                bending_energy = tf.reduce_sum([BendingEnergy.compute_displacement_energy(operand[..., i, :],
                                                                                          bending_energy_weight)
                                                for i in range(n_subject)])
                if regularization_operand == 'vector':
                    bending_energy *= 2 ** self.net.int_steps
                setattr(self.net, 'bending_energy', bending_energy)
                self.net.cost += bending_energy

            # add norm square
            if self.net.regularizer_type[3] == 'norm_square':
                with tf.name_scope('norm_square'):
                    norm_square = tf.reduce_sum(
                        tf.reduce_mean(operand ** 2, axis=(0, 1, 2, 3, -1))) * self.net.regularization_coefficient[3]
                    if regularization_operand == 'vector':
                        norm_square *= 2 ** self.net.int_steps
                    setattr(self.net, 'norm_square', norm_square)
                    self.net.cost += norm_square

        # compute number of negative Jacobians
        with tf.name_scope('negative_jacobians'):
            Energy = LocalDisplacementEnergy(energy_type=None)
            jacobian_det = tf.stack([Energy.compute_jacobian_determinant(self.net.ddf[..., i, :])
                                     for i in range(n_subject)])
            num_neg_jacob = tf.math.count_nonzero(tf.less_equal(jacobian_det, 0), dtype=tf.float32,
                                                  name='negative_jacobians_number')
            setattr(self.net, 'num_neg_jacob', num_neg_jacob)

        # initialize optimizer
        with tf.name_scope('optimizer'):
            self.optimizer, self.train_op, \
            self.init_lr, self.learning_rate_node = self._get_optimizer(self.net.cost, self.net.global_step,
                                                                        clip_gradient, lr=self.learning_rate,
                                                                        decay_steps=opt_decay_steps,
                                                                        step_size=2 * training_iters * self_iters,
                                                                        gamma=0.99998)
            if pretrain_epochs:
                _, self.pretrain_op, \
                _, self.pretrain_lr = self._get_optimizer(self.net.pretrain_cost,
                                                          global_step=tf.Variable(0, trainable=False, dtype=tf.int32),
                                                          optimizer='adam', lr=1e-4)

        # create a summary protocol buffer for learning rate
        with tf.name_scope('lr_summary'):
            tf.summary.scalar('learning_rate', self.learning_rate_node)

        # Merges summaries in the default graph
        self.summary_op = tf.summary.merge_all()

        # create an op that initializes all training variables
        init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

        self.prediction_path = prediction_path
        self.restore_model_path = restore_model_path
        self.save_model_path = save_model_path
        abs_prediction_path = os.path.abspath(prediction_path)
        abs_model_path = os.path.abspath(save_model_path)

        # remove the previous directory for model storing and validation prediction
        if not restore:
            self.net.logger.info("Removing '{:}'".format(abs_prediction_path))
            shutil.rmtree(abs_prediction_path, ignore_errors=True)
            self.net.logger.info("Removing '{:}'".format(abs_model_path))
            shutil.rmtree(abs_model_path, ignore_errors=True)

        # create a new directory for model storing and validation prediction
        if not os.path.exists(abs_prediction_path):
            self.net.logger.info("Allocating '{:}'".format(abs_prediction_path))
            os.makedirs(abs_prediction_path)

        if not os.path.exists(abs_model_path):
            self.net.logger.info("Allocating '{:}'".format(abs_model_path))
            os.makedirs(abs_model_path)

        return init

    def train(self, train_data_provider, test_data_provider, validation_batch_size, save_model_path, pretrain_epochs=5,
              epochs=100, dropout=0.2, clip_gradient=False, display_step=1, self_iters=1, decay_epochs=1,
              restore=False, write_graph=False, prediction_path='validation_prediction', restore_model_path=None,
              **kwargs):
        """
        Launch the training process.

        :param train_data_provider: Callable returning training data.
        :param test_data_provider: Callable returning validation data.
        :param validation_batch_size: The number of data for validation.
        :param save_model_path: The path where to store checkpoints.
        :param pretrain_epochs: Number of pre-training epochs.
        :param epochs: The number of epochs.
        :param dropout: The dropout probability.
        :param clip_gradient: Whether to apply gradient clipping.
        :param display_step: The number of steps till outputting stats.
        :param self_iters: The number of self iterations.
        :param decay_epochs: The number of epochs for learning rate decay.
        :param restore: Flag if previous model should be restored.
        :param restore_model_path: Where to restore the previous model.
        :param write_graph: Flag if the computation graph should be written as proto-buf file to the output path.
        :param prediction_path: The path where to save predictions on each epoch.
        """
        saver = tf.train.Saver(var_list=self.net.variables_to_restore, max_to_keep=kwargs.pop('max_to_keep', 5))
        best_saver = tf.train.Saver(var_list=self.net.variables_to_restore)
        save_path = os.path.join(save_model_path, "best_model.ckpt")
        # moving_average_path = os.path.join(save_model_path, "moving_average_model.ckpt")

        train_data_loader = DataLoader(train_data_provider, batch_size=self.batch_size, shuffle=True,
                                       num_workers=self.num_workers, collate_fn=train_data_provider.collate_fn)
        training_iters = len(train_data_loader)
        self.net.logger.info("Number of training iteration each epoch: %s" % training_iters)

        init = self._initialize(training_iters, self_iters, decay_epochs, epochs,
                                clip_gradient, restore_model_path, save_model_path,
                                restore, prediction_path, pretrain_epochs)

        # finalize default graph in case for inadvertently creation of new nodes
        # tf.get_default_graph().finalize()

        with tf.Session(config=config) as sess:
            if write_graph:
                tf.train.write_graph(sess.graph_def, save_model_path, "graph.pb", False)

            # initialization
            sess.run(init)

            # pre-training
            if pretrain_epochs:
                pretrain_saver = tf.train.Saver(var_list=self.net.variables_to_restore)
                pretrain_loss = 0.
                self.net.logger.info("Start pre-training by minimizing L2-norm of dense displacement fields......")
                for epoch in range(pretrain_epochs):
                    for step, batch in enumerate(train_data_loader):
                        _, loss, pretrain_lr = sess.run((self.pretrain_op, self.net.pretrain_cost, self.pretrain_lr),
                                                        feed_dict={self.net.data['target_image']: batch['target_image'],
                                                                   # self.net.data['target_label']: batch['target_label'],
                                                                   # self.net.data['atlases_label']: batch['atlases_label'],
                                                                   self.net.data['atlases_image']: batch[
                                                                       'atlases_image'],
                                                                   # self.net.pi: batch_pi,
                                                                   self.net.dropout_rate: dropout,
                                                                   self.net.train_phase: True})
                        pretrain_loss += loss

                        if step % display_step == 0:
                            self.net.logger.info("[Pre-training] Epoch: %d, Step: %d, "
                                                 "Pre-training loss: %.4f" % (epoch, step, loss))

                    self.net.logger.info("[Pre-training] Epoch: %d, "
                                         "Average pre-training loss: %.4f, "
                                         "Learning rate: %.2e" % (epoch, pretrain_loss / ((epoch + 1) * training_iters),
                                                                  pretrain_lr)
                                         )
                self.net.save(pretrain_saver, sess, os.path.join(save_model_path, 'pretrain_model.ckpt'),
                              latest_filename='pretrain_checkpoint')
                self.net.logger.info("Finish network pre-training!")

            # restore variables
            if restore:
                self.net.logger.info("Restoring from model path: %s" % restore_model_path)
                if '.ckpt' in restore_model_path:
                    self.net.logger.info("Restoring checkpoint: %s" % restore_model_path)
                    new_saver = tf.train.import_meta_graph(restore_model_path)
                    new_saver.restore(sess, restore_model_path, var_list=self.net.variables_to_restore)
                else:
                    ckpt = tf.train.get_checkpoint_state(restore_model_path,
                                                         latest_filename=kwargs.pop('latest_filename', None))
                    if ckpt and ckpt.model_checkpoint_path:
                        self.net.logger.info("Restoring checkpoint: %s" % ckpt.model_checkpoint_path)
                        self.net.restore(sess, ckpt.model_checkpoint_path, var_list=self.net.variables_to_restore)
                    else:
                        ckpt = tf.train.get_checkpoint_state(save_model_path,
                                                             latest_filename=kwargs.pop('latest_filename', None))
                        if ckpt and ckpt.model_checkpoint_path:
                            self.net.logger.info("Restoring checkpoint: %s" % ckpt.model_checkpoint_path)
                            self.net.restore(sess, ckpt.model_checkpoint_path, var_list=self.net.variables_to_restore)
                        else:
                            raise ValueError("Unknown previous model path: " % ckpt.model_checkpoint_path)

            # create summary writer for training summaries
            summary_writer = tf.summary.FileWriter(save_model_path, graph=sess.graph)

            # get validation metrics of epoch -1
            epoch_test_metrics = self.store_prediction(sess, test_data_provider, validation_batch_size,
                                                       dropout_rate=dropout, epoch=0)

            # create dictionary to record training/validation metrics for visualization
            test_metrics = {"Loss": {}, "Dice": {}, "Jaccard": {}, "Myocardial Dice": {},
                            "DDF norm": {}, "Bending energy": {}, "# Negative Jacobians": {}}
            train_metrics = {"Loss": {}, "Dice": {}, "Jaccard": {}, "Myocardial Dice": {},
                             "DDF norm": {}, "Bending energy": {}, "# Negative Jacobians": {}}

            # store epoch -1 metrics
            for k, v in test_metrics.items():
                v[0] = epoch_test_metrics[k]

            if epochs == 0:
                return save_path, train_metrics, test_metrics

            self.net.logger.info(
                "Start Unified Multi-Atlas Seg-Net optimization based on loss function: {}, prob_sigma: {} "
                "regularizer type: {} with regularization coefficient: {}, optimizer type: {}, "
                "batch size: {}, initial learning rate: {:.2e}".format(self.net.cost_name,
                                                                       self.net.prob_sigma,
                                                                       self.net.regularizer_type,
                                                                       self.net.regularization_coefficient,
                                                                       self.optimizer_name, self.batch_size,
                                                                       self.init_lr))

            lr = 0.
            assert self_iters >= 1
            total_loss = 0.
            for epoch in range(epochs):
                for step, batch in enumerate(train_data_loader):
                    # optimization operation (back-propagation)
                    # print(step)
                    for self_step in range(self_iters):
                        _, loss = sess.run((self.train_op, self.net.cost),
                                           feed_dict={self.net.data['target_image']: batch['target_image'],
                                                      self.net.data['target_label']: batch['target_label'],
                                                      self.net.data['target_weight']: batch['target_weight'],
                                                      self.net.data['atlases_label']: batch['atlases_label'],
                                                      self.net.data['atlases_image']: batch['atlases_image'],
                                                      self.net.data['atlases_weight']: batch['atlases_weight'],
                                                      self.net.dropout_rate: dropout,
                                                      self.net.train_phase: True})
                        total_loss += loss

                    # display mini-batch statistics and record training metrics
                    if step % display_step == 0:
                        # get training metrics for the display step
                        step_train_metrics, grads, lr = self.output_minibatch_stats(sess, summary_writer, epoch, step,
                                                                                    batch, dropout_rate=dropout)
                        # record training losses
                        for k, v in train_metrics.items():
                            v[epoch * training_iters * self_iters + (step + 1) * self_iters] = step_train_metrics[k]

                # display epoch statistics
                self.output_epoch_stats(epoch, total_loss, (epoch + 1) * training_iters * self_iters, lr)

                epoch_test_metrics = self.store_prediction(sess, test_data_provider, validation_batch_size,
                                                           dropout_rate=dropout, epoch=epoch+1)

                # save the current model if it is the best one hitherto
                if epoch > 0 and epoch_test_metrics['Dice'] >= np.max(list(test_metrics['Dice'].values())):
                    save_path = self.net.save(best_saver, sess, save_path, latest_filename='best_checkpoint')

                # save the current model
                self.net.save(saver, sess, os.path.join(save_model_path, 'epoch%s_model.ckpt' % epoch),
                              global_step=(epoch + 1) * training_iters * self_iters)
                # self.net.save(sess, moving_average_path, latest_filename='moving_average_checkpoint')

                # record epoch validation metrics
                for k, v in test_metrics.items():
                    v[(epoch + 1) * training_iters * self_iters] = epoch_test_metrics[k]

                # visualise training and validation metrics
                utils.visualise_metrics([train_metrics, test_metrics],
                                           save_path=os.path.dirname(self.prediction_path),
                                           labels=['training', 'validation'])

            self.net.logger.info("Optimization Finished!")
            self.net.save(saver, sess, os.path.join(save_model_path, 'checkpoint.ckpt'))

            return save_path, train_metrics, test_metrics

    def store_prediction(self, sess, test_data_provider, validation_batch_size, dropout_rate, **kwargs):
        """
        Compute validation metrics and store visualization results.

        :param sess: The pre-defined session for running TF operations.
        :param test_data_provider: The test data-provider.
        :param validation_batch_size: The validation data size.
        :param dropout_rate: The dropout probability.
        :param save_prefix: The save prefix for results saving.
        :return: A dictionary containing validation metrics.
        """
        save_prefix = kwargs.pop('save_prefix', '')
        epoch = kwargs.pop('epoch', '')
        loss = np.zeros([validation_batch_size])
        dice = np.zeros([validation_batch_size])
        jaccard = np.zeros([validation_batch_size])
        myo_dice = np.zeros([validation_batch_size])
        ddf_norm = np.zeros([validation_batch_size])
        bending_energy = np.zeros([validation_batch_size])
        num_neg_jacob = np.zeros([validation_batch_size])
        sess.run(tf.local_variables_initializer())
        # randomly sample the validation data at each epoch
        data_indices = random.sample(range(len(test_data_provider)), validation_batch_size)
        for i in range(validation_batch_size):
            data = test_data_provider[data_indices[i]]
            loss[i], dice[i], jaccard[i], myo_dice[i], ddf_norm[i], bending_energy[i], \
                num_neg_jacob[i], \
                test_pred, ddf = sess.run((self.net.cost, self.net.average_dice, self.net.jaccard,
                                            self.net.myocardial_dice, self.net.ddf_norm,
                                            self.net.bending_energy, self.net.num_neg_jacob,
                                            self.net.segmenter, self.net.ddf),
                                           feed_dict={self.net.data['target_image']: data['target_image'],
                                                      self.net.data['target_label']: data['target_label'],
                                                      self.net.data['target_weight']: data['target_weight'],
                                                      self.net.data['atlases_label']: data['atlases_label'],
                                                      self.net.data['atlases_image']: data['atlases_image'],
                                                      self.net.data['atlases_weight']: data['atlases_weight'],
                                                      # self.net.pi: test_pi[i],
                                                      self.net.train_phase: False,
                                                      self.net.dropout_rate: dropout_rate})

            utils.save_prediction_png(data['target_image'], data['target_label'], test_pred,
                                      os.path.join(self.prediction_path, 'epoch%s' % epoch), name_index=data_indices[i],
                                      data_provider=test_data_provider, save_prefix=save_prefix)

            utils.save_prediction_nii(test_pred.squeeze(0), os.path.join(self.prediction_path, 'epoch%s' % epoch),
                                      test_data_provider, name_index=data_indices[i], data_type='label',
                                      affine=data['target_affine'], header=data['target_header'],
                                      save_prefix=save_prefix)

            utils.save_prediction_nii(ddf.squeeze((0, -2)), os.path.join(self.prediction_path, 'epoch%s' % epoch),
                                      test_data_provider, name_index=data_indices[i], data_type='vector_fields',
                                      affine=data['target_affine'], header=data['target_header'],
                                      save_prefix=save_prefix)

        '''
        acc, auc, sens, spec = sess.run([self.net.acc[0], self.net.auc[0], self.net.sens[0], self.net.spec[0]])
        '''
        metrics = {'Loss': np.mean(loss), 'DDF norm': np.mean(ddf_norm), 'Dice': np.mean(dice),
                   'Jaccard': np.mean(jaccard), 'Myocardial Dice': np.mean(myo_dice),
                   'Bending energy': np.mean(bending_energy), '# Negative Jacobians': np.mean(num_neg_jacob)}

        self.net.logger.info("[Validation] Loss= {:.4f}, DDF norm= {:.4f}, "
                             "Bending energy= {:.3e}, # Negative Jacobians= {:.3e}, "
                             "Dice= {:.4f}, Jaccard= {:.4f}, "
                             "Myocardial Dice= {:.4f}".format(metrics['Loss'], metrics['DDF norm'],
                                                              metrics['Bending energy'],
                                                              metrics['# Negative Jacobians'],
                                                              metrics['Dice'], metrics['Jaccard'],
                                                              metrics['Myocardial Dice'])
                             )

        return metrics

    def output_epoch_stats(self, epoch, total_loss, training_iters, lr):
        self.net.logger.info(
            "[Training] Epoch {:}, Average Loss: {:.4f}, "
            "learning rate: {:.1e}".format(epoch, (total_loss / training_iters), lr))

    def output_minibatch_stats(self, sess, summary_writer, epoch, step, batch, dropout_rate):
        # Calculate batch loss and metrics
        sess.run(tf.local_variables_initializer())

        summary_str, \
        loss, lr, grads, \
        dice, jaccard, myo_dice, \
        ddf_norm, bending_energy, \
        num_neg_jacob = sess.run((self.summary_op, self.net.cost, self.learning_rate_node, self.net.gradients_node,
                                  self.net.average_dice, self.net.jaccard, self.net.myocardial_dice,
                                  self.net.ddf_norm, self.net.bending_energy, self.net.num_neg_jacob),
                                 feed_dict={self.net.data['target_image']: batch['target_image'],
                                            self.net.data['target_label']: batch['target_label'],
                                            self.net.data['target_weight']: batch['target_weight'],
                                            self.net.data['atlases_label']: batch['atlases_label'],
                                            self.net.data['atlases_image']: batch['atlases_image'],
                                            self.net.data['atlases_weight']: batch['atlases_weight'],
                                            # self.net.pi: batch_pi,
                                            self.net.train_phase: False,
                                            self.net.dropout_rate: dropout_rate})
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

        metrics = {'Loss': loss, 'DDF norm': ddf_norm, 'Dice': dice, 'Jaccard': jaccard,
                   'Myocardial Dice': myo_dice, "Bending energy": bending_energy,
                   "# Negative Jacobians": num_neg_jacob,
                   'Average gradient norm': np.mean([np.linalg.norm(g) for g in grads])}

        self.net.logger.info("[Training] Epoch {:}, Iteration {:}, Mini-batch Loss= {:.4f}, "
                             "Learning rate= {:.3e}, DDF norm= {:.4f}, Bending energy= {:.3e}, "
                             "# Negative Jacobians= {:.3e}, Average gradient norm {:.3e}, "
                             "Average foreground Dice= {:.4f}, "
                             "Myocardial Dice= {:.4f}".format(epoch, step, metrics['Loss'],
                                                              lr, metrics['DDF norm'],
                                                              metrics['Bending energy'],
                                                              metrics['# Negative Jacobians'],
                                                              metrics['Average gradient norm'],
                                                              metrics['Dice'], metrics['Myocardial Dice'])
                             )

        return metrics, grads, lr

    def __str__(self):
        # Todo: to make the print more complete
        return str(self.net) + '\n' \
                               "################ Training Setups ################\n" \
                               "batch_size= {}, optimizer_name= {}, num_workers= {}, \n" \
                               "initial learning_rate= {:.2f}, decay_rate= {:.2f}, \n" \
                               "restore_model_path= {}, saved_model_path= {}, prediction_path= {}, \n" \
                               "training_iters= {}, self_iters= {}, " \
                               "decay_epochs= {}".format(self.batch_size, self.optimizer_name,
                                                         self.num_workers, self.init_lr, self.decay_rate,
                                                         self.restore_model_path, self.save_model_path,
                                                         self.prediction_path, self.training_iters,
                                                         self.self_iters, self.decay_epochs)

    __repr__ = __str__


#####################################################################
# Helper functions
#####################################################################

def _compute_gradients(tensor, var_list):
    grads = tf.gradients(tensor, var_list)
    return [grad if grad is not None else tf.zeros_like(var)
            for var, grad in zip(var_list, grads)]
