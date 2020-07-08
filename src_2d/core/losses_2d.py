# -*- coding: utf-8 -*-
"""
Modules for loss construction.

@author: Xinzhe Luo
"""

import numpy as np
import tensorflow as tf
from tqdm import tqdm
from numba import jit
from core import utils_2d

config = tf.ConfigProto(device_count={'GPU': 0})  # cpu only


class DiceLoss(object):
    """
    Compute the multi-scale Dice loss between the ground truth and the prediction.

    """

    def __init__(self, eps=1., dice_type='multiclass', scales=(0, 1), **kwargs):

        self.eps = eps
        self.dice_type = dice_type
        self.scales = scales
        self.kwargs = kwargs
        assert (self.dice_type in ['multiclass', 'binary']), "Dice type has to be 'generalised' or 'binary'!"

    def loss(self, target_prob, warped_atlases_prob):
        """
        :param target_prob: target probability map/one-hot label of shape [n_batch, *vol_shape, n_class]
        :param warped_atlases_prob: atlas probability map of shape [n_batch, *vol_shape, n_class]
        """
        y_true = target_prob
        y_pred = warped_atlases_prob

        n_dims = y_true.shape.ndims
        if self.dice_type == 'multiclass':
            numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=list(range(n_dims - 1)))
            denominator = tf.reduce_sum(y_true ** 2 + y_pred ** 2, axis=list(range(n_dims - 1)))
            dice = tf.reduce_mean(numerator / tf.maximum(denominator, self.eps))

        elif self.dice_type == 'binary':
            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true ** 2 + y_pred ** 2)
            dice = tf.reduce_mean(numerator / tf.maximum(denominator, self.eps))

        return tf.subtract(1., dice, name='dice_loss')

    def multi_scale_loss(self, target_prob, warped_atlases_probs):
        """
        :param target_prob: target probability map of shape [n_batch, *vol_shape, n_class]
        :param warped_atlases_probs: list of tensors of shape [n_batch, *vol_shape, n_class], each one computed from a
            certain scale of atlas probability map
        """
        assert len(warped_atlases_probs) == len(self.scales), "The number of warped atlases by multi-level ddfs " \
                                                                  "should be equal to the number of scales!"
        multi_scale_losses = []
        for prob in warped_atlases_probs:
            multi_scale_losses.append(self.loss(target_prob, prob))
        multi_scale_losses = tf.stack(multi_scale_losses)

        return tf.reduce_mean(multi_scale_losses, name='multi_scale_dice_loss')


class CrossEntropy(object):
    """
    Compute cross entropy loss from two probability maps.

    """
    def __init__(self, eps=1e-8, **kwargs):
        self.eps = eps
        self.kwargs = kwargs
        self.reduce_mean = kwargs.pop('reduce_mean', True)

    def loss(self, y_true, y_pred):
        """
        compute loss

        :param y_true: target probability map of shape [n_batch, *vol_shape, n_class]
        :param y_pred: prediction probability map of shape [n_batch, *vol_shape, n_class]
        :return:
        """
        y_pred = tf.clip_by_value(y_pred, self.eps, 1.)
        cross_entropy = - tf.reduce_sum(y_true * tf.log(y_pred), axis=-1)
        if self.reduce_mean:
            cross_entropy = tf.reduce_mean(cross_entropy)
        return cross_entropy



class LabelConsistencyLoss(object):
    """
    Compute multi-scale label consistency loss between the target and atlas probability maps.
    """
    def __init__(self, eps=1e-5, scales=(0, 1,), **kwargs):
        self.eps = eps
        self.scales = scales
        self.method = kwargs.pop('prob_method', 'average')
        self.bk_axis = kwargs.pop('bk_axis', 0)
        self.kwargs = kwargs

    def loss(self, target_prob, warped_atlases_prob, prior):
        """
        Return the label consistency loss.
        :param target_prob: target probability map of shape [n_batch, *vol_shape, n_class]
        :param warped_atlases_prob: atlas probability map of shape [n_batch, *vol_shape, n_class]
        :return: the label consistency loss
        """
        label_const = target_prob * warped_atlases_prob * prior
        log_likelihood = tf.log(tf.clip_by_value(tf.reduce_sum(label_const, axis=-1), self.eps, 1.),
                                name='log_likelihood')
        if self.method == 'use_mask':
            # prob_threshold = self.kwargs.pop('prob_threshold', (0.025, 0.975))  # probability thresholds to compute mask
            # assert prob_threshold[0] < prob_threshold[1]
            # mask = ((warped_atlases_prob[..., self.bk_axis] < prob_threshold[1]) &
            #        (warped_atlases_prob[..., self.bk_axis] > prob_threshold[0])) | \
            #        ((target_prob[..., self.bk_axis] < prob_threshold[1]) &
            #        (target_prob[..., self.bk_axis] > prob_threshold[0]))
            # # mask = tf.logical_not(tf.equal(warped_atlases_prob[..., self.bk_axis],
            # #                                target_prob[..., self.bk_axis]), name='prob_mask')
            mask = tf.cast(utils_2d.compute_mask_from_prob(target_prob), tf.bool) | tf.cast(utils_2d.compute_mask_from_prob(warped_atlases_prob), tf.bool)
            mask = tf.cast(tf.reduce_any(mask, axis=-1), tf.float32)
            return tf.negative(tf.divide(tf.reduce_sum(log_likelihood * mask), tf.reduce_sum(mask) + self.eps),
                               name='label_consistency_loss_with_mask')
        elif self.method == 'use_roi':
            vol_shape = target_prob.get_shape().as_list()[1:-1]
            mag_rate = self.kwargs.pop('mag_rate', 0.1)
            foreground_flag = tf.reduce_any(tf.concat([target_prob[..., 1:] > 0,
                                                       warped_atlases_prob[..., 1:] > 0], axis=-1), axis=(0, -1),
                                            name='foreground_flag')  # [*vol_shape], union on all samples within a batch
            arg_index = tf.where(foreground_flag, name='arg_index')  # [num_true, 3], same among the mini-batch
            low = tf.cast(tf.reduce_min(arg_index, axis=0), dtype=tf.float32)
            high = tf.cast(tf.reduce_max(arg_index, axis=0), dtype=tf.float32)
            soft_low = tf.maximum(tf.floor(low - (high - low) * mag_rate / 2), tf.zeros_like(low, dtype=tf.int32))  # [3]
            soft_high = tf.minimum(tf.floor(high + (high - low) * mag_rate / 2), tf.constant(vol_shape, tf.int32))  # [3]

            return tf.negative(tf.reduce_mean(log_likelihood[:,
                                                             soft_low[0]:soft_high[0],
                                                             soft_low[1]:soft_high[1],
                                                             soft_low[2]:soft_high[2], ]),
                               name='label_consistency_loss_with_roi')
        elif self.method == 'average':
            return tf.negative(tf.reduce_mean(log_likelihood), name='average_label_consistency_loss')

        elif self.method == 'sum':
            return tf.negative(tf.reduce_sum(log_likelihood), name='sum_label_consistency_loss')

        elif self.method == 'use_prob':
            raise NotImplementedError

        else:
            raise ValueError("Unknown cost method: %s" % self.method)

    def multi_scale_loss(self, target_prob, warped_atlases_probs, prior):
        """
        :param target_prob: target probability map of shape [n_batch, *vol_shape, n_class]
        :param warped_atlases_probs: list of tensors of shape [n_batch, *vol_shape, n_class], each one computed from a
            certain scale of atlas probability map
        """
        assert len(warped_atlases_probs) == len(self.scales), "The number of warped atlases by multi-level ddfs " \
                                                              "should be equal to the number of scales!"
        multi_scale_losses = []
        for prob in warped_atlases_probs:
            multi_scale_losses.append(self.loss(target_prob, prob, prior))
        multi_scale_losses = tf.stack(multi_scale_losses)

        return tf.reduce_mean(multi_scale_losses, name='multi_scale_label_consistency')


class KLDivergenceLoss(object):
    """
    Compute the KL-divergence loss between the true posterior and approximate posterior, only applicable to binary
    and single-atlas segmentation currently.

    Todo: Generalise the class to multi-class and multi-atlas segmentation.

    """
    def __init__(self, eps=1e-10, beta=0.5, r=2, **kwargs):
        """
        Initialization.

        :param eps: epsilon for probability truncation
        :param beta: parameter to control the weight distribution
        :param r: radius of the cube-shaped neighbourhood
        """
        self.eps = eps
        self.beta = beta
        self.r = r

    def loss(self, target_image, atlases_image, atlases_prob):
        """
        Compute the KL-divergence loss.

        :param target_image: target image tensor of shape [n_batch, *vol_shape, channels]
        :param atlases_image: warped atlases image tensor of shape [n_batch, *vol_shape, n_atlas, channels]
        :param atlases_prob: warped atlases prob tensor of shape [n_batch, *vol_shape, n_atlas, 2]
        :return: loss
        """
        assert atlases_prob.get_shape().as_list()[-1] == 2, "Only applicable to binary segmentation in current version!"
        assert atlases_prob.get_shape().as_list()[-2] == 1, "Only applicable to single-atlas segmentation in current version!"
        atlases_prob = tf.clip_by_value(atlases_prob, self.eps, 1-self.eps)
        target_image = tf.reduce_sum(target_image, -1, keepdims=True)
        # print(target_image.shape.as_list())
        atlases_image = tf.reduce_sum(tf.squeeze(atlases_image, -2), -1, keepdims=True)
        # print(atlases_image)
        loss = tf.reduce_mean(tf.reduce_sum(atlases_prob * tf.log(atlases_prob), axis=-1))
        diff_image = tf.subtract(target_image, atlases_image)
        # print(diff_image.shape)
        dist_weight = tf.clip_by_value(tf.pow(utils_2d.separable_filter3d(tf.square(diff_image),
                                                                          tf.constant([1]*(self.r*2+1), tf.float32)),
                                              self.beta), self.eps, 1-self.eps)

        loss -= tf.reduce_mean(tf.reduce_sum(atlases_prob * tf.concat([dist_weight, 1-dist_weight], axis=-1), axis=-1))
        return loss


class MvMMNetLoss(object):
    """
    Compute the MvMM-Net loss.

    """
    def __init__(self, eps=1e-5, **kwargs):
        self.eps = eps
        self.kwargs = kwargs

    def loss_weight(self, target_prob, atlases_prob, target_weight, atlases_weight, prior):
        """
        Compute the MvMM-Net loss function.

        :param target_prob: target probability map of shape [n_batch, *vol_shape, n_class]
        :param atlases_prob: warped atlases probability map of shape [n_batch, *vol_shape, n_atlas, n_class]
        :param target_weight: target weight map of shape [n_batch, *vol_shape, n_class]
        :param atlases_weight: atlases weight map of shape [n_batch, *vol_shape, n_atlas, n_class]
        :param prior: prior probability of shape [1, 1, 1, n_class]
        :return:
        """
        with tf.name_scope('mvmm_net_loss_weight'):
            # n_atlas = atlases_image.get_shape().as_list()[-2]
            # assert n_atlas == atlases_prob.get_shape().as_list()[-2]
            # product of probability maps
            prob_product = utils_2d.get_joint_prob(atlases_prob) * target_prob * prior
            # product of weight maps
            weight_product = tf.reduce_prod(utils_2d.get_normalized_prob(atlases_weight),
                                            axis=-2) * utils_2d.get_normalized_prob(target_weight)
            weight_product = tf.stop_gradient(weight_product)
            # compute log-likelihood
            all_product = weight_product * prob_product
            sum_mask = tf.cast(tf.reduce_any(all_product > self.eps, axis=-1), tf.float32)
            ll = tf.log(tf.clip_by_value(tf.reduce_sum(all_product, axis=-1), self.eps, 1.))

            return tf.negative(tf.divide(tf.reduce_sum(ll*sum_mask), tf.reduce_sum(sum_mask)+self.eps))

    def loss_mask(self, target_prob, atlases_prob, prior):
        """
        Compute the MvMM loss function.

        :param target_prob: target probability map of shape [n_batch, *vol_shape, n_class]
        :param atlases_prob: warped atlases probability map of shape [n_batch, *vol_shape, n_atlas, n_class]
        :param prior: a tensor of shape [1, 1, 1, n_class]
        :return: loss function
        """
        with tf.name_scope('mvmm_net_loss_mask'):
            n_atlas = atlases_prob.get_shape().as_list()[-2]
            # product of probability maps
            prob_product = utils_2d.get_joint_prob(atlases_prob) * target_prob * prior
            # product of masks
            target_mask = utils_2d.compute_mask_from_prob(target_prob)
            atlases_mask = tf.stack([utils_2d.compute_mask_from_prob(atlases_prob[..., i, :])
                                     for i in range(n_atlas)], axis=-2)
            mask_product = tf.reduce_prod(utils_2d.get_normalized_prob(atlases_mask),
                                          axis=-2) * utils_2d.get_normalized_prob(target_mask)
            # compute log-likelihood
            all_product = mask_product * prob_product
            sum_mask = tf.cast(tf.reduce_any(all_product > self.eps, axis=-1), tf.float32)

            ll = tf.log(tf.clip_by_value(tf.reduce_sum(all_product, axis=-1), self.eps, 1.))

            return tf.negative(tf.divide(tf.reduce_sum(ll*sum_mask), tf.reduce_sum(sum_mask)+self.eps))


class CrossCorrelation(object):
    """
    Compute local (normalized) cross correlation.

    """
    def __init__(self, win=7, eps=1e-5, **kwargs):
        self.win = win if isinstance(win, (list, tuple)) else [win, ]*2
        self.eps = eps
        self.kwargs = kwargs
        self.beta = kwargs.pop('beta', 0.5)
        self.kernel = kwargs.pop('kernel', 'ones')

    def ncc(self, target, source):
        """
        Compute local normalized cross correlation between target and source intensities.

        :param target: target tensor of shape [n_batch, *vol_shape, 1]
        :param source: source tensor of shape [n_batch, *vol_shape, 1]
        :return: cross correlation of shape [n_batch, *vol_shape, 1]
        """
        with tf.name_scope('cross_correlation'):
            # compute CC squares
            T2 = tf.square(target)
            S2 = tf.square(source)
            TS = tf.multiply(target, source)

            # compute filters and local sums
            if self.kernel == 'ones':
                kernel = tf.ones([*self.win, 1, 1])
                strides = [1, 1, 1, 1]
                T_sum = tf.nn.conv2d(target, kernel, strides, padding='SAME')
                S_sum = tf.nn.conv2d(source, kernel, strides, padding='SAME')
                T2_sum = tf.nn.conv2d(T2, kernel, strides, padding='SAME')
                S2_sum = tf.nn.conv2d(S2, kernel, strides, padding='SAME')
                TS_sum = tf.nn.conv2d(TS, kernel, strides, padding='SAME')
            elif self.kernel == 'gaussian':
                kernel = utils_2d.gauss_kernel1d(self.win[0] // 6)
                T_sum = utils_2d.separable_filter2d(target, kernel)
                S_sum = utils_2d.separable_filter2d(source, kernel)
                T2_sum = utils_2d.separable_filter2d(T2, kernel)
                S2_sum = utils_2d.separable_filter2d(S2, kernel)
                TS_sum = utils_2d.separable_filter2d(TS, kernel)
            else:
                raise NotImplementedError

            # compute cross correlation
            win_size = np.prod(self.win)
            u_T = T_sum / win_size
            u_S = S_sum / win_size

            cor = TS_sum - u_S*T_sum - u_T*S_sum + u_T*u_S*win_size
            T_var = T2_sum - 2*u_T*T_sum + u_T*u_T*win_size
            S_var = S2_sum - 2*u_S*S_sum + u_S*u_S*win_size

            return tf.pow((cor*cor) / (T_var*S_var + self.eps), self.beta)

    def loss(self, target, source):
        return tf.subtract(1., tf.reduce_mean(self.ncc(target, source)))


class LocalDisplacementEnergy(object):
    """
    Compute displacement energy as regularization.

    """
    def __init__(self, energy_type, **kwargs):
        self.energy_type = energy_type
        self.kwargs = kwargs
        self.mode = kwargs.pop('mode', 'tf')

    @staticmethod
    def _gradient_dx(fv):
        return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2

    @staticmethod
    def _gradient_dy(fv):
        return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2

    def _get_shape(self, x, axis=-1):
        if self.mode == 'tf':
            return x.get_shape().as_list()[axis]
        elif self.mode == 'np':
            return x.shape[axis]
        else:
            raise NotImplementedError

    def _gradient_txyz(self, Txyz, fn):
        return tf.stack([fn(Txyz[..., i]) for i in range(self._get_shape(Txyz, axis=-1))], axis=-1)

    def compute_displacement_energy(self, ddf, energy_weight):
        with tf.name_scope('displacement_energy'):
            self._dTdx = self._gradient_txyz(ddf, self._gradient_dx)  # [batch, *vol_shape, 3]
            self._dTdy = self._gradient_txyz(ddf, self._gradient_dy)
            if energy_weight == 0:
                return tf.constant(0.0)

            if energy_weight is not None:
                if self.energy_type == 'bending':
                    dTdxx = self._gradient_txyz(self._dTdx, self._gradient_dx)
                    dTdyy = self._gradient_txyz(self._dTdy, self._gradient_dy)
                    dTdxy = self._gradient_txyz(self._dTdx, self._gradient_dy)
                    energy = tf.reduce_mean(dTdxx ** 2 + dTdyy ** 2 + 2 * dTdxy ** 2)

                elif self.energy_type == 'gradient-l2':
                    norms = self._dTdx ** 2 + self._dTdy ** 2 + self._dTdz ** 2
                    energy = tf.reduce_mean(norms)

                elif self.energy_type == 'gradient-l1':
                    norms = tf.abs(self._dTdx) + tf.abs(self._dTdy) + tf.abs(self._dTdz)
                    energy = tf.reduce_mean(norms)

                else:
                    raise NotImplementedError

                return tf.multiply(energy, energy_weight, name='displacement_energy')

    def compute_jacobian_determinant(self, ddf):
        """
        Compute the average Jacobian determinants of the displacement fields.

        :param ddf: The displacement fields of shape [batch, nx, ny, nz, 3].
        :return: The Jacobian determinant of the vector fields of shape [batch, nx, ny, nz].
        """
        with tf.name_scope('jacobian_determinant'):
            if not hasattr(self, '_dTdx'):
                self._dTdx = self._gradient_txyz(ddf, self._gradient_dx)  # [batch, *vol_shape, 3]
            if not hasattr(self, '_dTdy'):
                self._dTdy = self._gradient_txyz(ddf, self._gradient_dy)

            jacobian_det = tf.subtract((self._dTdx[..., 0] + 1) * (self._dTdy[..., 1] + 1),
                                       self._dTdx[..., 1] * self._dTdy[..., 0],
                                       name='jacobian_det')
            return jacobian_det


class MutualInformation(object):
    """
    Compute mutual-information-based metrics.

    """

    def __init__(self, n_bins=64, sigma=3, **kwargs):
        self.n_bins = n_bins
        self.sigma = 2*sigma**2
        self.kwargs = kwargs
        self.eps = kwargs.pop('eps', 1e-10)
        self.win = kwargs.pop('win', 7); assert self.win % 2 == 1  # window size for local metrics
        self._normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
        self._normalizer_2d = 2.0 * np.pi * sigma ** 2
        self.background_method = kwargs.pop('background_method', 'min')
        if self.background_method is None:
            self.background_value = kwargs.pop('background_value')

    def _compute_marginal_entropy(self, values, bins):
        """
        Compute the marginal entropy using Parzen window estimation.

        :param values: a tensor of shape [n_batch, *vol_shape, channels]
        :param bins: a tensor of shape [n_bins, 1]
        :return: entropy - the marginal entropy;
                 p - the probability distribution
        """
        p = np.exp(-(np.square(np.reshape(np.mean(values, axis=-1), [-1]) - bins) / self.sigma)) / self._normalizer_1d
        p_norm = np.mean(p, axis=1)
        p_norm = p_norm / (np.sum(p_norm) + self.eps)
        entropy = - np.sum(p_norm * np.log2(p_norm + self.eps))

        return entropy, p

    def mi(self, target, source):
        """
        Compute mutual information: I(target, source) = H(target) + H(source) - H(target, source).

        :param target:
        :param source:
        :return:
        """
        if self.background_method == 'min':
            background_fixed = np.min(target)
            background_moving = np.min(source)
        elif self.background_method == 'mean':
            background_fixed = np.mean(target)
            background_moving = np.mean(source)
        elif self.background_method is None:
            background_fixed = self.background_value
            background_moving = self.background_value
        else:
            raise NotImplementedError

        bins_target = np.expand_dims(np.linspace(background_fixed, np.max(target), self.n_bins), axis=-1)
        bins_source = np.expand_dims(np.linspace(background_moving, np.max(source), self.n_bins), axis=-1)

        # TODO: add masks

        # Compute marginal entropy
        entropy_target, p_t = self._compute_marginal_entropy(target, bins_target)
        entropy_source, p_s = self._compute_marginal_entropy(source, bins_source)

        # compute joint entropy
        p_joint = np.matmul(p_t, p_s.transpose(1, 0)) / self._normalizer_2d
        p_joint = p_joint / (np.sum(p_joint) + self.eps)

        entropy_joint = - np.sum(p_joint * np.log2(p_joint + self.eps))

        return entropy_target + entropy_source - entropy_joint

    def nmi(self, target, source):
        """
        Compute normalized mutual information: NMI(target, source) = (H(target) + H(source)) / H(target, source).

        :param target:
        :param source:
        :return:
        """
        if self.background_method == 'min':
            background_fixed = np.min(target)
            background_moving = np.min(source)
        elif self.background_method == 'mean':
            background_fixed = np.mean(target)
            background_moving = np.mean(source)
        elif self.background_method is None:
            background_fixed = self.background_value
            background_moving = self.background_value
        else:
            raise NotImplementedError

        bins_target = np.expand_dims(np.linspace(background_fixed, np.max(target), self.n_bins), axis=-1)
        bins_source = np.expand_dims(np.linspace(background_moving, np.max(source), self.n_bins), axis=-1)

        # TODO: add masks

        # Compute marginal entropy
        entropy_target, p_t = self._compute_marginal_entropy(target, bins_target)
        entropy_source, p_s = self._compute_marginal_entropy(source, bins_source)

        # compute joint entropy
        p_joint = np.matmul(p_t, p_s.transpose(1, 0)) / self._normalizer_2d
        p_joint = p_joint / (np.sum(p_joint) + self.eps)

        entropy_joint = - np.sum(p_joint * np.log2(p_joint + self.eps))

        return (entropy_target + entropy_source) / (entropy_joint + self.eps)

    def ecc(self, target, source):
        """
        Compute entropy correlation coefficient: ECC(target, source) = 2 - 2 / NMI(target, source).

        :param target:
        :param source:
        :return:
        """

        return 2 - 2 / (self.nmi(target, source) + self.eps)

    def ce(self, target, source):
        """
        Compute conditional entropy: H(target|source) = H(target, source) - H(source).

        :param target:
        :param source:
        :return:
        """
        if self.background_method == 'min':
            background_fixed = np.min(target)
            background_moving = np.min(source)
        elif self.background_method == 'mean':
            background_fixed = np.mean(target)
            background_moving = np.mean(source)
        elif self.background_method is None:
            background_fixed = self.background_value
            background_moving = self.background_value
        else:
            raise NotImplementedError

        bins_target = np.expand_dims(np.linspace(background_fixed, np.max(target), self.n_bins), axis=-1)
        bins_source = np.expand_dims(np.linspace(background_moving, np.max(source), self.n_bins), axis=-1)

        # TODO: add masks

        # Compute marginal entropy
        entropy_target, p_t = self._compute_marginal_entropy(target, bins_target)
        entropy_source, p_s = self._compute_marginal_entropy(source, bins_source)

        # compute joint entropy
        p_joint = np.matmul(p_t, p_s.transpose(1, 0)) / self._normalizer_2d
        p_joint = p_joint / (np.sum(p_joint) + self.eps)

        entropy_joint = - np.sum(p_joint * np.log2(p_joint + self.eps))

        return entropy_joint - entropy_source

    def lce(self, target, source):
        """
        Compute local conditional entropy

        :param target: target tensor of shape [n_batch, *vol_shape, 1]
        :param source: source tensor of shape [n_batch, *vol_shape, 1]
        :return: local conditional entropy of shape [n_batch, *vol_shape, 1]
        """
        batch, nx, ny, nz = target.shape[0:4]
        lce = np.zeros_like(target)
        padding = self.win // 2
        target_pad = np.pad(target, pad_width=((0, 0), (padding, padding), (padding, padding),
                                               (padding, padding), (0, 0)), mode='constant')
        source_pad = np.pad(source, pad_width=((0, 0), (padding, padding), (padding, padding),
                                               (padding, padding), (0, 0)), mode='constant')

        with tqdm(total=lce.size) as pbar:
            for n in range(batch):
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            lce[n, i, j, k] = self.ce(
                                self._normalize(target_pad[n, i:(i+self.win), j:(j+self.win), k:(k+self.win)]),
                                self._normalize(source_pad[n, i:(i+self.win), j:(j+self.win), k:(k+self.win)]))
                            pbar.update(1)
        return lce

    def lmi(self, target, source):
        """
        Compute local mutual information

        :param target: target tensor of shape [n_batch, *vol_shape, 1]
        :param source: source tensor of shape [n_batch, *vol_shape, 1]
        :return: local conditional entropy of shape [n_batch, *vol_shape, 1]
        """
        batch, nx, ny, nz = target.shape[0:4]
        lmi = np.zeros_like(target)
        padding = self.win // 2
        target_pad = np.pad(target, pad_width=((0, 0), (padding, padding), (padding, padding),
                                               (padding, padding), (0, 0)), mode='constant')
        source_pad = np.pad(source, pad_width=((0, 0), (padding, padding), (padding, padding),
                                               (padding, padding), (0, 0)), mode='constant')

        with tqdm(total=lmi.size) as pbar:
            for n in range(batch):
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            lmi[n, i, j, k] = self.mi(
                                self._normalize(target_pad[n, i:(i + self.win), j:(j + self.win), k:(k + self.win)]),
                                self._normalize(source_pad[n, i:(i + self.win), j:(j + self.win), k:(k + self.win)]))
                            pbar.update(1)
        return lmi

    def lnmi(self, target, source):
        """
        Compute local normalized mutual information

        :param target: target tensor of shape [n_batch, *vol_shape, 1]
        :param source: source tensor of shape [n_batch, *vol_shape, 1]
        :return: local conditional entropy of shape [n_batch, *vol_shape, 1]
        """
        batch, nx, ny, nz = target.shape[0:4]
        lnmi = np.zeros_like(target)
        padding = self.win // 2
        target_pad = np.pad(target, pad_width=((0, 0), (padding, padding), (padding, padding),
                                               (padding, padding), (0, 0)), mode='constant')
        source_pad = np.pad(source, pad_width=((0, 0), (padding, padding), (padding, padding),
                                               (padding, padding), (0, 0)), mode='constant')

        with tqdm(total=lnmi.size) as pbar:
            for n in range(batch):
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            lnmi[n, i, j, k] = self.nmi(
                                self._normalize(target_pad[n, i:(i + self.win), j:(j + self.win), k:(k + self.win)]),
                                self._normalize(source_pad[n, i:(i + self.win), j:(j + self.win), k:(k + self.win)]))
                            pbar.update(1)
        return lnmi

    def lecc(self, target, source):
        """
        Compute local entropy correlation coefficient

        :param target: target tensor of shape [n_batch, *vol_shape, 1]
        :param source: source tensor of shape [n_batch, *vol_shape, 1]
        :return: local conditional entropy of shape [n_batch, *vol_shape, 1]
        """
        batch, nx, ny, nz = target.shape[0:4]
        lecc = np.zeros_like(target)
        padding = self.win // 2
        target_pad = np.pad(target, pad_width=((0, 0), (padding, padding), (padding, padding),
                                               (padding, padding), (0, 0)), mode='constant')
        source_pad = np.pad(source, pad_width=((0, 0), (padding, padding), (padding, padding),
                                               (padding, padding), (0, 0)), mode='constant')

        with tqdm(total=lecc.size) as pbar:
            for n in range(batch):
                for i in range(nx):
                    for j in range(ny):
                        for k in range(nz):
                            lecc[n, i, j, k] = self.ecc(
                                self._normalize(target_pad[n, i:(i + self.win), j:(j + self.win), k:(k + self.win)]),
                                self._normalize(source_pad[n, i:(i + self.win), j:(j + self.win), k:(k + self.win)]))
                            pbar.update(1)
        return lecc

    def _normalize(self, data):
        data -= data.min()
        data /= (data.max() + self.eps)
        return data


# Numba accelerated computation

@jit
def _lecc(target, source, n_bins, win, sigma):
    """
    Compute local entropy correlation coefficient

    :param target: target tensor of shape [n_batch, *vol_shape, 1]
    :param source: source tensor of shape [n_batch, *vol_shape, 1]
    :return: local conditional entropy of shape [n_batch, *vol_shape, 1]
    """
    batch, nx, ny, nz = target.shape[0:4]
    lecc = np.zeros_like(target)
    padding = win // 2
    target_pad = np.pad(target, pad_width=((0, 0), (padding, padding), (padding, padding),
                                           (padding, padding), (0, 0)), mode='constant')
    source_pad = np.pad(source, pad_width=((0, 0), (padding, padding), (padding, padding),
                                           (padding, padding), (0, 0)), mode='constant')

    for n in range(batch):
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    lecc[n, i, j, k] = _ecc(_normalize(target_pad[n, i:(i + win), j:(j + win), k:(k + win)]),
                                            _normalize(source_pad[n, i:(i + win), j:(j + win), k:(k + win)]),
                                            n_bins, sigma)

    return lecc


@jit(nopython=True)
def _normalize(data):
    data -= np.min(data)
    data /= (np.max(data) + 1e-10)
    return data


@jit
def _nmi(target, source, n_bins, sigma):
    """
    Compute normalized mutual information: NMI(target, source) = (H(target) + H(source)) / H(target, source).

    :param target:
    :param source:
    :return:
    """
    bins_target = np.expand_dims(np.linspace(np.min(target), np.max(target), n_bins), axis=-1)
    bins_source = np.expand_dims(np.linspace(np.min(source), np.max(source), n_bins), axis=-1)

    # TODO: add masks

    # Compute marginal entropy
    entropy_target, p_t = _compute_marginal_entropy(target, bins_target, sigma)
    entropy_source, p_s = _compute_marginal_entropy(source, bins_source, sigma)

    # compute joint entropy
    p_joint = np.matmul(p_t, p_s.transpose(1, 0)) / _normalizer_2d(sigma)
    p_joint = p_joint / (np.sum(p_joint) + 1e-10)

    entropy_joint = - np.sum(p_joint * np.log2(p_joint + 1e-10))

    return (entropy_target + entropy_source) / (entropy_joint + 1e-10)


@jit
def _ecc(target, source, n_bins, sigma):
    """
    Compute entropy correlation coefficient: ECC(target, source) = 2 - 2 / NMI(target, source).

    :param target:
    :param source:
    :return:
    """

    return 2 - 2 / (_nmi(target, source, n_bins, sigma) + 1e-10)


@jit
def _compute_marginal_entropy(values, bins, sigma):
    """
    Compute the marginal entropy using Parzen window estimation.

    :param values: a tensor of shape [n_batch, *vol_shape, channels]
    :param bins: a tensor of shape [n_bins, 1]
    :return: entropy - the marginal entropy;
             p - the probability distribution
    """
    p = np.exp(-(np.square(np.reshape(np.mean(values, axis=-1), [-1]) - bins) / sigma)) / _normalizer_1d(sigma)
    p_norm = np.mean(p, axis=1)
    p_norm = p_norm / (np.sum(p_norm) + 1e-10)
    entropy = - np.sum(p_norm * np.log2(p_norm + 1e-10))

    return entropy, p


@jit(nopython=True)
def _normalizer_1d(sigma):
    return np.sqrt(2.0 * np.pi) * sigma


@jit(nopython=True)
def _normalizer_2d(sigma):
    return 2.0 * np.pi * sigma ** 2