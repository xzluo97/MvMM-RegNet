# -*- coding: utf-8 -*-
"""
Functions and operations for performance visualization and result store,
some of which are not used in the current situation.

@author: Xinzhe Luo
"""
from __future__ import print_function, division, absolute_import, unicode_literals


import numpy as np
import os
import logging
from PIL import Image
import nibabel as nib
import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import re
import itertools
import math
from sklearn import mixture
from scipy import stats, signal
from core import metrics_2d

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True

# from sklearn.metrics import roc_curve, auc

##############################################################
# functions for output saving and visualisation
##############################################################

def to_rgb(img):
    """
    Converts the given array into a RGB image. If the number of channels is not
    3 the array is tiled such that it has 3 channels. Finally, the values are
    rescaled to [0,255)

    :param img: the array to convert [n, *vol_shape, channels]

    :returns img: the rgb image [n, *vol_shape, 3]
    """
    if len(img.shape) < 5:
        img = np.expand_dims(img, axis=-1)
    channels = img.shape[-1]
    if channels < 3:
        img = np.tile(img, 3)

    img[np.isnan(img)] = 0
    for k in range(np.shape(img)[3]):
        st = img[:, :, :, k, ]
        if np.amin(st) != np.amax(st):
            st -= np.amin(st)
            st /= np.amax(st)
        st *= 255
    return img.round().astype(np.uint8)


def combine_img_prediction(data, gt, pred):
    """
    Combines the data, ground truth and the prediction into one rgb image for each class

    :param data: the data tensor, of shape [1, *vol_shape, 1]
    :param gt: the ground truth tensor, of shape [1, *vol_shape, n_class]
    :param pred: the prediction tensor, of shape [1, *vol_shape, n_class]
    :param slice_indices: List of slice indices for visualization.
    :param slice_axis: The slicing axis.

    :returns: the concatenated rgb image
    """

    n_class = gt.shape[-1]
    class_labels = []
    class_preds = []
    for k in range(n_class):
        class_labels.append(dye_label(to_rgb(crop_to_shape(gt[..., k], pred.shape)), class_index=k))
        class_preds.append(dye_label(to_rgb(pred[..., k]), class_index=k))

    # print(np.sum(np.stack(class_labels), axis=0).shape)
    # print(image_column.shape)

    image_column = to_rgb(crop_to_shape(data.squeeze(-1), pred.shape))
    label_column = cv2.addWeighted(np.sum(np.stack(class_labels), axis=0).astype(np.uint8), 0.5, image_column, 0.7, 0)
    pred_column = cv2.addWeighted(np.sum(np.stack(class_preds), axis=0).astype(np.uint8), 0.5, image_column, 0.7, 0)

    final = np.squeeze(np.concatenate([image_column, label_column, pred_column], axis=2), axis=0)
    return final


def dye_label(label, class_index):
    """
    Dye the label with colors.

    :param label: The RGB one-hot label of shape [1, *vol_shape, 3].
    :param class_index: The class index.
    :return: The colorized label map.
    Todo:
        Enable customized RGB values.
    """
    rgb_values = [[0, 0, 0], [255, 250, 205], [188, 143, 143], [199, 21, 133],
                  [188, 143, 143], [135, 206, 235], [238, 130, 238], [253, 245, 230]]

    label[..., 0][label[..., 0] == 255] = rgb_values[class_index][0]
    label[..., 1][label[..., 1] == 255] = rgb_values[class_index][1]
    label[..., 2][label[..., 2] == 255] = rgb_values[class_index][2]

    return label


def save_image(imgs, path):
    """
    Writes the image to disk

    :param imgs: the rgb images to save
    :param path: the target path
    """
    for i in range(len(imgs)):
        img_path = os.path.join(os.path.split(path)[0], 'class%d_' % i + os.path.split(path)[1])
        Image.fromarray(imgs[i].round().astype(np.uint8)).save(img_path, 'PNG', dpi=[300, 300], quality=95)


def save_prediction_png(image, label, pred, save_path, data_provider, **kwargs):
    """
    Combine each prediction and the corresponding ground truth as well as input into one image and save as png files.

    :param image: The raw image array.
    :param label: The one-hot ground-truth array.
    :param pred: The prediction array.
    :param save_path: where to save the validation/test predictions, has the form of 'directory'
    :param image_names: The names of the original data.
    :param data_provider: An instance of the test data provider.
    :param kwargs: name_index - The name index of the target-atlas image pair if given;
                   save_name - The saved filename if given;
                   save_prefix - The prefix of the saved filename;
                   slice_indices - List of slice indices for visualization.
    """
    save_name = kwargs.pop("save_name", None)
    save_prefix = kwargs.pop("save_prefix", '')

    abs_pred_path = os.path.abspath(save_path)
    if not os.path.exists(abs_pred_path):
        logging.info("Allocating '{:}'".format(abs_pred_path))
        os.makedirs(abs_pred_path)

    if save_name is None:
        name_index = kwargs.pop("name_index")
        target_name, atlases_name = data_provider.get_image_names(name_index)
        t_name = '_'.join(os.path.basename(target_name).split('_')[0:3])
        a_names = '+'.join(['_'.join(os.path.basename(a_name).split('_')[0:3]) for a_name in atlases_name])
        save_name = 'target-' + t_name + '_atlas-' + a_names

    pred = np.where(np.equal(np.max(pred, -1, keepdims=True), pred),
                    np.ones_like(pred), np.zeros_like(pred))

    save_suffix = 'seg.png'

    plt.imsave(os.path.join(save_path, save_prefix + '_'.join([save_name, save_suffix])),
               combine_img_prediction(image, label, pred))


def save_prediction_nii(pred, save_path, data_provider, data_type='image', **kwargs):
    """
    Save the predictions into nifty images.
    Predictions are pre-processed by setting the maximum along classes to be a certain intensity specific to its class.

    :param pred: The prediction array of shape [*vol_shape, n_class/channels].
    :param affine: The affine matrix array that relates array coordinates from the image data array to coordinates
        in some RAS+ world coordinate system.
    :param header: The header that contains the image metadata.
    :param save_path: where to save the validation/test predictions, has the form of 'directory'
    :param data_provider: The data provider for model prediction.
    :param save_suffix: the suffix of the saved file
    :param data_provider: An instance of the test data provider.
    :param kwargs: name_index - The name index of the target-atlas image pair if given;
                   save_name - The saved filename if given;
                   header - The header that contains the image metadata;
                   save_prefix - The prefix of the saved filename;
                   save_suffix - The suffix of the saved filename
                   original_size - The original size of the input for padding

    """
    save_name = kwargs.pop("save_name", None)
    affine = kwargs.pop("affine", np.eye(4))
    header = kwargs.pop("header", None)
    save_prefix = kwargs.pop("save_prefix", '')
    original_size = kwargs.pop("original_size", None)

    if len(pred.shape) < 4:
        pred = np.expand_dims(pred, -2)

    if original_size is None:
        original_size = pred.shape

    abs_pred_path = os.path.abspath(save_path)
    if not os.path.exists(abs_pred_path):
        logging.info("Allocating '{:}'".format(abs_pred_path))
        os.makedirs(abs_pred_path)

    if save_name is None:
        name_index = kwargs.pop("name_index")
        target_name, atlases_name = data_provider.get_image_names(name_index)
        t_name = '_'.join(os.path.basename(target_name).split('_')[0:3])
        a_names = '+'.join(['_'.join(os.path.basename(a_name).split('_')[0:3]) for a_name in atlases_name])
        save_name = 'target-' + t_name + '_atlas-' + a_names

    if data_type == 'image':
        save_suffix = kwargs.pop("save_suffix", 'image.nii.gz')
        save_dtype = kwargs.pop("save_dtype", np.uint16)
        squeeze_channel = kwargs.pop("squeeze_channel", True)
        pred_pad = np.pad(pred,
                          (((original_size[0] - pred.shape[0]) // 2,
                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                           ((original_size[1] - pred.shape[1]) // 2,
                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                           (0, 0), (0, 0)), 'constant')
        if squeeze_channel:
            pred_pad = np.mean(pred_pad, axis=-1)
        img = nib.Nifti1Image(pred_pad.astype(save_dtype), affine=affine, header=header)
        nib.save(img, os.path.join(save_path, save_prefix + '_'.join([save_name, save_suffix])))

    elif data_type == 'vector_fields':
        save_suffix = kwargs.pop("save_suffix", 'vector.nii.gz')
        save_dtype = kwargs.pop("save_dtype", np.float32)
        if pred.shape[-1] <= 2:
            zero_fields = np.zeros([pred.shape[0], pred.shape[1], pred.shape[2], 3 - pred.shape[-1]])
            pred = np.concatenate([pred, zero_fields], axis=-1)
        pred_pad = np.pad(pred,
                          (((original_size[0] - pred.shape[0]) // 2,
                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                           ((original_size[1] - pred.shape[1]) // 2,
                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                           (0, 0), (0, 0)), 'constant')
        img = nib.Nifti1Image(pred_pad.astype(save_dtype), affine=affine, header=header)
        nib.save(img, os.path.join(save_path, save_prefix + '_'.join([save_name, save_suffix])))

    elif data_type == 'label':
        save_suffix = kwargs.pop("save_suffix", 'seg.nii.gz')
        save_dtype = kwargs.pop("save_dtype", np.uint16)
        class_preds = []
        for i in range(pred.shape[-1]):
            if i == 0:
                class_preds.append(np.pad(pred[..., i],
                                          (((original_size[0] - pred.shape[0]) // 2,
                                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                                           ((original_size[1] - pred.shape[1]) // 2,
                                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                                           (0, 0)), 'constant', constant_values=1))
            else:
                class_preds.append(np.pad(pred[..., i],
                                          (((original_size[0] - pred.shape[0]) // 2,
                                            original_size[0] - pred.shape[0] - (original_size[0] - pred.shape[0]) // 2),
                                           ((original_size[1] - pred.shape[1]) // 2,
                                            original_size[1] - pred.shape[1] - (original_size[1] - pred.shape[1]) // 2),
                                           (0, 0)), 'constant'))

        pred = np.stack(class_preds, -1)

        intensity = np.tile(np.asarray(data_provider.label_intensity), np.concatenate((pred.shape[:-1], [1])))
        mask = np.equal(np.max(pred, -1, keepdims=True), pred)
        img = nib.Nifti1Image(np.sum(mask * intensity, axis=-1).astype(save_dtype), affine=affine, header=header)
        nib.save(img, os.path.join(save_path, save_prefix + '_'.join([save_name, save_suffix])))


def save_prediction_numpy(predictions, save_path, image_names, image_suffix, save_prefix):
    """
    Save the predictions into numpy array.

    :param predictions: list of predictions
    :param save_path: where to save the validation/test predictions, has the form of 'directory'
    :param image_names: The names of the original data.
    :param image_suffix: The name suffix of the original data.
    :param save_prefix: The name prefix of the file to save.
    """
    abs_pred_path = os.path.abspath(save_path)
    if not os.path.exists(abs_pred_path):
        logging.info("Allocating '{:}'".format(abs_pred_path))
        os.makedirs(abs_pred_path)

    for k in range(len(predictions)):
        name = os.path.split(image_names[k])[-1]
        pred = predictions[k]
        np.save(os.path.join(save_path, save_prefix + name.replace(image_suffix, '_seg.npy')), pred)


############################################################
# functions for metrics visulisation
############################################################

def visualise_metrics(metrics, save_path, **kwargs):
    """
    Visualise training and test metrics for comparison.

    :param metrics: A list of metrics from different datasets for visualisation.
    :param save_path: Where to save the figure.
    :return: None.
    """
    if not isinstance(metrics, (list, tuple)):
        metrics = list(metrics)

    assert len(metrics) <= 9, "Number of types of metrics should be less than 8."
    assert checkEqual([m.keys() for m in metrics]), "Types of metrics must be equal among all datasets."

    labels = kwargs.pop('labels', ['dataset %s' % k for k in range(len(metrics))])
    linewidth = kwargs.pop('linewidth', 1)
    if isinstance(linewidth, (int, float)):
        linewidth = [linewidth] * len(metrics)

    # markersize = kwargs.pop('markersize', 12)
    # if isinstance(markersize, (int, float)):
    #     markersize = [markersize] * len(metrics)

    metric_types = list(metrics[0].keys())
    n_rows, n_cols = factor_int(len(metric_types))
    fig, ax = plt.subplots(n_rows, n_cols, squeeze=False, figsize=(12 * n_cols, 6 * n_rows))
    fig.canvas.set_window_title('Visualisation of metrics from various datasets.')

    plt.plot()
    colors = ['b','k', 'gray', 'g', 'c', 'm', 'orange', 'yellow', 'r']
    # markers = ['.', 'o', 'v', '^', '<', '>', '+', 'x', '*']

    for i in range(n_rows):
        for j in range(n_cols):
            type = metric_types[i * n_cols + j]
            max_x = np.max(list(itertools.chain(*[list(m[type].keys()) for m in metrics])))
            max_y = np.max(list(itertools.chain(*[list(m[type].values()) for m in metrics])))
            min_y = np.min(list(itertools.chain(*[list(m[type].values()) for m in metrics])))

            for k in range(len(metrics)):
                m = metrics[k]
                # extract data
                x, y = list(m[type].keys()), list(m[type].values())

                # plot metrics
                ax[i, j].plot(x, y, linestyle='-', color=colors[k], label=labels[k], linewidth=linewidth[k],
                              # marker=markers[k], markersize=markersize[k]
                              )

            # set title
            ax[i, j].set_title(type)
            # set label
            ax[i, j].set_xlabel('Iterations')
            ax[i, j].set_ylabel('Values')
            # set grid
            ax[i, j].grid(b='true', which='both', axis='both', color='gray')
            # set limits
            ax[i, j].set_xlim(-1, max_x * 1.1)
            ax[i, j].set_ylim(min(min_y * 1.1, 0), max_y * 1.1)

            # set legend
            ax[i, j].legend(loc='upper left')

    save_name = kwargs.pop('save_name', "metrics_visualisation.png")
    plt.savefig(os.path.join(save_path, save_name), bbox_inches='tight', dpi=300)


def factor_int(n):
    """
    Factorize an integer into factors as close to a squared root as possible.

    :param n: The given integer.
    :return: The two factors.
    """
    nsqrt = math.ceil(math.sqrt(n))
    solution = False
    val = nsqrt
    while not solution:
        val2 = int(n / val)
        if val2 * val == float(n):
            solution = True
        else:
            val -= 1
    return val, val2


def checkEqual(lst):
    return lst[1:] == lst[:-1]


############################################################
# functions for tensor manipulation
############################################################

'''
def crop_to_shape(data, shape):
    """
    Crops the array to the given image shape by removing the border 
    (expects a tensor of shape [batches, nx, ny, channels].

    :param data: the array to crop, shape=[n, *vol_shape, n_class]
    :param shape: the target shape
    """
    assert np.all(data.shape>=shape), "The shape of array to be cropped is smaller than the target shape."
    offset1 = (data.shape[1] - shape[1])//2
    offset2 = (data.shape[2] - shape[2])//2
    offset3 = (data.shape[3] - shape[3])//2

    return data[:, offset1:(offset1+shape[1]), offset2:(offset2+shape[2]), offset3:(offset3+shape[3]), :]
'''


def crop_to_shape(data, shape, mode='np'):
    """
    Crops the volumetric tensor or array into the given image shape by removing the border
    (expects a tensor or array of shape [n_batch, *vol_shape, channels]).

    :param data: the tensor or array to crop, shape=[n_batch, *vol_shape, n_class]
    :param shape: the target shape
    :param mode: 'np' or 'tf'.
    :return: The cropped tensor or array.
    """
    assert mode in ['np', 'tf'], "The mode must be either 'np' or 'tf'!"
    if shape is None:
        return data

    if mode == 'np':
        data_shape = data.shape
    elif mode == 'tf':
        data_shape = data.get_shape().as_list()
    else:
        raise NotImplementedError

    if len(shape) <= 2:
        shape = (1, ) + shape + (data_shape[-1], )

    assert np.all(tuple(data_shape[1:3]) >= shape[1:3]), "The shape of array to be cropped is smaller than the " \
                                                         "target shape."
    offset0 = (data_shape[1] - shape[1]) // 2
    offset1 = (data_shape[2] - shape[2]) // 2
    remainder0 = (data_shape[1] - shape[1]) % 2
    remainder1 = (data_shape[2] - shape[2]) % 2

    if (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) == 0:
        return data

    elif (data_shape[1] - shape[1]) != 0 and (data_shape[2] - shape[2]) == 0:
        return data[:, offset0:(-offset0 - remainder0), ]

    elif (data_shape[1] - shape[1]) == 0 and (data_shape[2] - shape[2]) != 0:
        return data[:, :, offset1:(-offset1 - remainder1), ]

    else:
        return data[:, offset0:(-offset0 - remainder0), offset1:(-offset1 - remainder1)]


def pad_to_shape_image(data, shape, **kwargs):
    """
    Pad the array to the given shape by the edge values

    :param data: The data for padding, of shape [n_batch, *vol_shape, n_class].
    :param shape: The shape of the padded array, of value [*vol_shape].
    """
    mode = kwargs.pop('mode', 'tf')
    method = kwargs.pop('method', 'constant')
    offset1 = (shape[0] - data.shape[1]) // 2
    offset2 = (shape[1] - data.shape[2]) // 2
    remainder1 = (shape[0] - data.shape[1]) % 2
    remainder2 = (shape[1] - data.shape[2]) % 2

    if mode == 'tf':
        assert np.all(data.get_shape().as_list()[1:4] <= list(shape)), "The shape of array to be padded is larger than the target shape."
        return tf.pad(data, ((0, 0), (offset1, offset1 + remainder1), (offset2, offset2 + remainder2), (0, 0)),
                      mode=method, **kwargs)

    elif mode == 'np':
        assert np.all(data.shape[1:4] <= tuple(shape)), "The shape of array to be padded is larger than the target shape."
        return np.pad(data, ((0, 0),
                             (offset1, offset1 + remainder1),
                             (offset2, offset2 + remainder2),
                             (0, 0)), mode=method, **kwargs)

    else:
        raise NotImplementedError


def pad_to_shape_label(label, shape):
    """
    Pad the label array to the given shape by 0 and 1.

    :param label: The label for padding, of shape [n_batch, *vol_shape, n_class].
    :param shape: The shape of the padded array, of value [n_batch, *vol_shape, n_class].
    :return: The padded label array.
    """
    assert np.all(label.shape <= shape), "The shape of array to be padded is larger than the target shape."
    offset1 = (shape[1] - label.shape[1]) // 2
    offset2 = (shape[2] - label.shape[2]) // 2
    remainder1 = (shape[1] - label.shape[1]) % 2
    remainder2 = (shape[2] - label.shape[2]) % 2

    class_pred = []
    for k in range(label.shape[-1]):
        if k == 0:
            class_pred.append(np.pad(label[..., k],
                                     ((0, 0),
                                      (offset1, offset1 + remainder1),
                                      (offset2, offset2 + remainder2)),
                                     'constant', constant_values=1))

        else:
            class_pred.append(np.pad(label[..., k],
                                     ((0, 0),
                                      (offset1, offset1 + remainder1),
                                      (offset2, offset2 + remainder2)),
                                     'constant'))

    return np.stack(class_pred, axis=-1)


###################################################
# functions for label fusion
###################################################

def majority_voting(predictions):
    """
    Get the binary segmentation masks by majority voting.

    :param predictions: A list containing predictions of probability/segmentation maps, each of shape [1, *vol_shape, n_class].
    :return: An array representing the segmentation map after majority voting.
    """
    # n = len(predictions)
    # masks = []
    # for i in range(n):
    #     masks.append(np.where(np.equal(np.max(predictions[i], -1, keepdims=True), predictions[i]),
    #                           np.ones_like(predictions[i]),
    #                           np.zeros_like(predictions[i])))

    votes = np.sum(np.stack(predictions), axis=0)  # [1, *vol_shape, n_class]

    return np.where(np.equal(np.max(votes, -1, keepdims=True), votes),
                    np.ones_like(votes),
                    np.zeros_like(votes))


###########################################################
# functions for image re-sampling
###########################################################

def interpn(vol, loc, interp_method='linear'):
    """
    N-D gridded interpolation in tensorflow

    vol can have more dimensions than loc[i], in which case loc[i] acts as a slice
    for the first dimensions

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc: a N-long list of N-D Tensors (the interpolation locations) for the new grid
            each tensor has to have the same size (but not nec. same size as vol)
            or a tensor of size [*new_vol_shape, D]
        interp_method: interpolation type 'linear' (default) or 'nearest'

    Returns:
        new interpolated volume of the same size as the entries in loc

    TODO:
        enable optional orig_grid - the original grid points.
        check out tf.contrib.resampler, only seems to work for 2D data
    """

    if isinstance(loc, (list, tuple)):
        loc = tf.stack(loc, -1)

    # since loc can be a list, nb_dims has to be based on vol.
    nb_dims = loc.shape[-1]

    if nb_dims != len(vol.shape[:-1]):
        raise Exception("Number of loc Tensors %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape[:-1])))

    if nb_dims > len(vol.shape):
        raise Exception("Loc dimension %d does not match volume dimension %d"
                        % (nb_dims, len(vol.shape)))

    if len(vol.shape) == nb_dims:
        vol = tf.expand_dims(vol, -1)

    # flatten and float location Tensors
    loc = tf.cast(loc, 'float32')

    if isinstance(vol.shape, (tf.Dimension, tf.TensorShape)):
        volshape = vol.shape.as_list()
    else:
        volshape = vol.shape

    # interpolate
    if interp_method == 'linear':
        loc0 = tf.floor(loc)

        # clip values
        max_loc = [d - 1 for d in vol.get_shape().as_list()]
        clipped_loc = [tf.clip_by_value(loc[..., d], 0, max_loc[d]) for d in range(nb_dims)]
        loc0lst = [tf.clip_by_value(loc0[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get other end of point cube
        loc1 = [tf.clip_by_value(loc0lst[d] + 1, 0, max_loc[d]) for d in range(nb_dims)]
        locs = [[tf.cast(f, 'int32') for f in loc0lst], [tf.cast(f, 'int32') for f in loc1]]

        # compute the difference between the upper value and the original value
        # differences are basically 1 - (pt - floor(pt))
        #   because: floor(pt) + 1 - pt = 1 + (floor(pt) - pt) = 1 - (pt - floor(pt))
        diff_loc1 = [loc1[d] - clipped_loc[d] for d in range(nb_dims)]
        diff_loc0 = [1 - d for d in diff_loc1]
        weights_loc = [diff_loc1, diff_loc0]  # note reverse ordering since weights are inverse of diff.

        # go through all the cube corners, indexed by a ND binary vector
        # e.g. [0, 0] means this "first" corner in a 2-D "cube"
        cube_pts = list(itertools.product([0, 1], repeat=nb_dims))
        interp_vol = 0

        for c in cube_pts:
            # get nd values
            # note re: indices above volumes via https://github.com/tensorflow/tensorflow/issues/15091
            #   It works on GPU because we do not perform index validation checking on GPU -- it's too
            #   expensive. Instead we fill the output with zero for the corresponding value. The CPU
            #   version caught the bad index and returned the appropriate error.
            subs = [locs[c[d]][d] for d in range(nb_dims)]

            # tf stacking is slow for large volumes, so we will use sub2ind and use single indexing.
            # indices = tf.stack(subs, axis=-1)
            # vol_val = tf.gather_nd(vol, indices)
            # faster way to gather than gather_nd, because the latter needs tf.stack which is slow :(
            idx = sub2ind(vol.shape[:-1], subs)
            vol_val = tf.gather(tf.reshape(vol, [-1, volshape[-1]]), idx)

            # get the weight of this cube_pt based on the distance
            # if c[d] is 0 --> want weight = 1 - (pt - floor[pt]) = diff_loc1
            # if c[d] is 1 --> want weight = pt - floor[pt] = diff_loc0
            wts_lst = [weights_loc[c[d]][d] for d in range(nb_dims)]
            # tf stacking is slow, we we will use prod_n()
            # wlm = tf.stack(wts_lst, axis=0)
            # wt = tf.reduce_prod(wlm, axis=0)
            wt = prod_n(wts_lst)
            wt = tf.expand_dims(wt, -1)

            # compute final weighted value for each cube corner
            interp_vol += wt * vol_val

    else:
        assert interp_method == 'nearest'
        roundloc = tf.cast(tf.round(loc), 'int32')

        # clip values
        max_loc = [tf.cast(d - 1, 'int32') for d in vol.shape]
        roundloc = [tf.clip_by_value(roundloc[..., d], 0, max_loc[d]) for d in range(nb_dims)]

        # get values
        # tf stacking is slow. replace with gather
        # roundloc = tf.stack(roundloc, axis=-1)
        # interp_vol = tf.gather_nd(vol, roundloc)
        idx = sub2ind(vol.shape[:-1], roundloc)
        interp_vol = tf.gather(tf.reshape(vol, [-1, vol.shape[-1]]), idx)

    return interp_vol


def resize(vol, zoom_factor, interp_method='linear'):
    """
    if zoom_factor is a list, it will determine the ndims, in which case vol has to be of length ndims of ndims + 1

    if zoom_factor is an integer, then vol must be of length ndims + 1

    """

    if isinstance(zoom_factor, (list, tuple)):
        ndims = len(zoom_factor)
        vol_shape = vol.shape[:ndims]

        assert len(vol_shape) in (ndims, ndims + 1), \
            "zoom_factor length %d does not match ndims %d" % (len(vol_shape), ndims)

    else:
        vol_shape = vol.shape[:-1]
        ndims = len(vol_shape)
        zoom_factor = [zoom_factor] * ndims
    if not isinstance(vol_shape[0], int):
        vol_shape = vol_shape.as_list()

    new_shape = [vol_shape[f] * zoom_factor[f] for f in range(ndims)]
    new_shape = [int(f) for f in new_shape]

    # get grid for new shape
    grid = volshape_to_ndgrid(new_shape)
    grid = [tf.cast(f, 'float32') for f in grid]
    offset = [grid[f] / zoom_factor[f] - grid[f] for f in range(ndims)]
    offset = tf.stack(offset, ndims)

    # transform
    return transform(vol, offset, interp_method)


def affine_to_shift(affine_matrix, volshape, shift_center=True, indexing='ij'):
    """
    transform an affine matrix to a dense location shift tensor in tensorflow

    Algorithm:
        - get grid and shift grid to be centered at the center of the image (optionally)
        - apply affine matrix to each index.
        - subtract grid

    Parameters:
        affine_matrix: ND+1 x ND+1 or ND x ND+1 matrix (Tensor)
        volshape: 1xN Nd Tensor of the size of the volume.
        shift_center (optional)

    Returns:
        shift field (Tensor) of size *volshape x N

    TODO:
        allow affine_matrix to be a vector of size nb_dims * (nb_dims + 1)
    """

    if isinstance(volshape, (tf.Dimension, tf.TensorShape)):
        volshape = volshape.as_list()

    if affine_matrix.dtype != 'float32':
        affine_matrix = tf.cast(affine_matrix, 'float32')

    nb_dims = len(volshape)

    if len(affine_matrix.shape) == 1:
        if len(affine_matrix) != (nb_dims * (nb_dims + 1)):
            raise ValueError('transform is supposed a vector of len ndims * (ndims + 1).'
                             'Got len %d' % len(affine_matrix))

        affine_matrix = tf.reshape(affine_matrix, [nb_dims, nb_dims + 1])

    if not (affine_matrix.shape[0] in [nb_dims, nb_dims + 1] and affine_matrix.shape[1] == (nb_dims + 1)):
        raise Exception('Affine matrix shape should match'
                        '%d+1 x %d+1 or ' % (nb_dims, nb_dims) +
                        '%d x %d+1.' % (nb_dims, nb_dims) +
                        'Got: ' + str(volshape))

    # list of volume ndgrid
    # N-long list, each entry of shape volshape
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)
    mesh = [tf.cast(f, 'float32') for f in mesh]

    if shift_center:
        mesh = [mesh[f] - (volshape[f] - 1) / 2 for f in range(len(volshape))]

    # add an all-ones entry and transform into a large matrix
    flat_mesh = [flatten(f) for f in mesh]
    flat_mesh.append(tf.ones(flat_mesh[0].shape, dtype='float32'))
    mesh_matrix = tf.transpose(tf.stack(flat_mesh, axis=1))  # 4 x nb_voxels

    # compute locations
    loc_matrix = tf.matmul(affine_matrix, mesh_matrix)  # N+1 x nb_voxels
    loc_matrix = tf.transpose(loc_matrix[:nb_dims, :])  # nb_voxels x N
    loc = tf.reshape(loc_matrix, list(volshape) + [nb_dims])  # *volshape x N
    # loc = [loc[..., f] for f in range(nb_dims)]  # N-long list, each entry of shape volshape

    # get shifts and return
    return loc - tf.stack(mesh, axis=nb_dims)


def transform(vol, loc_shift, interp_method='linear', indexing='ij'):
    """
    transform (interpolation N-D volumes (features) given shifts at each location in tensorflow

    Essentially interpolates volume vol at locations determined by loc_shift.
    This is a spatial transform in the sense that at location [x] we now have the data from,
    [x + shift] so we've moved data.

    Parameters:
        vol: volume with size vol_shape or [*vol_shape, nb_features]
        loc_shift: shift volume [*new_vol_shape, N]
        interp_method (default:'linear'): 'linear', 'nearest'
        indexing (default: 'ij'): 'ij' (matrix) or 'xy' (cartesian).
            In general, prefer to leave this 'ij'

    Return:
        new interpolated volumes in the same size as loc_shift[0]

    Keyworks:
        interpolation, sampler, resampler, linear, bilinear
    """

    # parse shapes
    if isinstance(loc_shift.shape, (tf.Dimension, tf.TensorShape)):
        volshape = loc_shift.shape[:-1].as_list()
    else:
        volshape = loc_shift.shape[:-1]
    nb_dims = len(volshape)

    # location should be mesh and delta
    mesh = volshape_to_meshgrid(volshape, indexing=indexing)  # volume mesh
    loc = [tf.cast(mesh[d], 'float32') + loc_shift[..., d] for d in range(nb_dims)]

    # test single
    return interpn(vol, loc, interp_method=interp_method)


def integrate_vec(vec, int_steps=1, **kwargs):
    """
    Integrate stationary vector fields by scaling and squaring.

    :param vec: the vector fields to be integrated, of shape [n_batch, *vol_shape, n_atlas, vol_dim]
    :param int_steps: number of integration times= 2**int_steps
    :return:
    """
    def integrate(vec):
        """
        :param vec: vector fields of shape [*vol_shape, n_atlas, vol_dim]
        :return:
        """
        n_atlas = vec.get_shape().as_list()[-2]
        int_vec = [vec[..., i, :] for i in range(n_atlas)]
        for i in range(n_atlas):
            for _ in range(int_steps):
                int_vec[i] += transform(int_vec[i], int_vec[i])
        return tf.stack(int_vec, axis=-2)

    # with tf.control_dependencies([tf.assert_less(tf.reduce_max(vec), eps)]):
    return tf.map_fn(integrate, vec, dtype=tf.float32)


def volshape_to_ndgrid(volshape, **kwargs):
    """
    compute Tensor ndgrid from a volume size

    Parameters:
        volshape: the volume size
        **args: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return ndgrid(*linvec, **kwargs)


def volshape_to_meshgrid(volshape, **kwargs):
    """
    compute Tensor meshgrid from a volume size

    Parameters:
        volshape: the volume size
        **args: "name" (optional)

    Returns:
        A list of Tensors

    See Also:
        tf.meshgrid, meshgrid, ndgrid, volshape_to_ndgrid
    """

    isint = [float(d).is_integer() for d in volshape]
    if not all(isint):
        raise ValueError("volshape needs to be a list of integers")

    linvec = [tf.range(0, d) for d in volshape]
    return meshgrid(*linvec, **kwargs)


def ndgrid(*args, **kwargs):
    """
    broadcast Tensors on an N-D grid with ij indexing
    uses meshgrid with ij indexing

    Parameters:
        *args: Tensors with rank 1
        **args: "name" (optional)

    Returns:
        A list of Tensors

    """
    return meshgrid(*args, indexing='ij', **kwargs)


def flatten(v):
    """
    flatten Tensor v

    Parameters:
        v: Tensor to be flattened

    Returns:
        flat Tensor
    """

    return tf.reshape(v, [-1])


def meshgrid(*args, **kwargs):
    """

    meshgrid code that builds on (copies) tensorflow's meshgrid but dramatically
    improves runtime by changing the last step to tiling instead of multiplication.
    https://github.com/tensorflow/tensorflow/blob/c19e29306ce1777456b2dbb3a14f511edf7883a8/tensorflow/python/ops/array_ops.py#L1921

    Broadcasts parameters for evaluation on an N-D grid.
    Given N one-dimensional coordinate arrays `*args`, returns a list `outputs`
    of N-D coordinate arrays for evaluating expressions on an N-D grid.
    Notes:
    `meshgrid` supports cartesian ('xy') and matrix ('ij') indexing conventions.
    When the `indexing` argument is set to 'xy' (the default), the broadcasting
    instructions for the first two dimensions are swapped.
    Examples:
    Calling `X, Y = meshgrid(x, y)` with the tensors
    ```python
    x = [1, 2, 3]
    y = [4, 5, 6]
    X, Y = meshgrid(x, y)
    # X = [[1, 2, 3],
    #      [1, 2, 3],
    #      [1, 2, 3]]
    # Y = [[4, 4, 4],
    #      [5, 5, 5],
    #      [6, 6, 6]]
    ```
    Args:
    *args: `Tensor`s with rank 1.
    **kwargs:
      - indexing: Either 'xy' or 'ij' (optional, default: 'xy').
      - name: A name for the operation (optional).
    Returns:
    outputs: A list of N `Tensor`s with rank N.
    Raises:
    TypeError: When no keyword arguments (kwargs) are passed.
    ValueError: When indexing keyword argument is not one of `xy` or `ij`.
    """

    indexing = kwargs.pop("indexing", "xy")
    name = kwargs.pop("name", "meshgrid")
    if kwargs:
        key = list(kwargs.keys())[0]
        raise TypeError("'{}' is an invalid keyword argument "
                        "for this function".format(key))

    if indexing not in ("xy", "ij"):
        raise ValueError("indexing parameter must be either 'xy' or 'ij'")

    # with ops.name_scope(name, "meshgrid", args) as name:
    ndim = len(args)
    s0 = (1,) * ndim

    # Prepare reshape by inserting dimensions with size 1 where needed
    output = []
    for i, x in enumerate(args):
        output.append(tf.reshape(tf.stack(x), (s0[:i] + (-1,) + s0[i + 1::])))
    # Create parameters for broadcasting each tensor to the full size
    shapes = [tf.size(x) for x in args]
    sz = [x.get_shape().as_list()[0] for x in args]

    # output_dtype = tf.convert_to_tensor(args[0]).dtype.base_dtype

    if indexing == "xy" and ndim > 1:
        output[0] = tf.reshape(output[0], (1, -1) + (1,) * (ndim - 2))
        output[1] = tf.reshape(output[1], (-1, 1) + (1,) * (ndim - 2))
        shapes[0], shapes[1] = shapes[1], shapes[0]
        sz[0], sz[1] = sz[1], sz[0]

    # This is the part of the implementation from tf that is slow.
    # We replace it below to get a ~6x speedup (essentially using tile instead of * tf.ones())
    # TODO(nolivia): improve performance with a broadcast
    # mult_fact = tf.ones(shapes, output_dtype)
    # return [x * mult_fact for x in output]
    for i in range(len(output)):
        output[i] = tf.tile(output[i], tf.stack([*sz[:i], 1, *sz[(i + 1):]]))
    return output


def prod_n(lst):
    prod = lst[0]
    for p in lst[1:]:
        prod *= p
    return prod


def sub2ind(siz, subs, **kwargs):
    """
    assumes column-order major

    referring to https://github.com/voxelmorph/voxelmorph/blob/master/ext/neuron/neuron/utils.py

    :param siz: A list indicating the dimension of each sampling coordinate.
    :param subs: A list of sub-coordinates indicating the sampling coordinate along each axis.
    """
    # subs is a list
    assert len(siz) == len(subs), 'found inconsistent siz and subs: %d %d' % (len(siz), len(subs))

    k = np.cumprod(siz[::-1])

    ndx = subs[-1]
    for i, v in enumerate(subs[:-1][::-1]):
        ndx = ndx + v * k[i]

    return ndx


###############################################################
# functions for loss construction
###############################################################

def get_predictor(logits):
    """
    produce the probability maps from the final feature maps of the network
    """
    return tf.nn.softmax(logits, axis=-1, name='probability_map')


def get_segmentation(predictor, mode='tf'):
    """
    produce the segmentation maps from the probability maps
    """
    assert mode in ['tf', 'np'], "The mode must be either 'tf' or 'np'!"
    if mode == 'tf':
        assert isinstance(predictor, tf.Tensor)
        return tf.where(tf.equal(tf.reduce_max(predictor, -1, keepdims=True), predictor),
                        tf.ones_like(predictor),
                        tf.zeros_like(predictor))

    elif mode == 'np':
        assert isinstance(predictor, np.ndarray)
        return np.where(np.equal(np.max(predictor, -1, keepdims=True), predictor),
                        np.ones_like(predictor),
                        np.zeros_like(predictor))


def get_joint_prob(prob, mode='tf', **kwargs):
    """
    Compute joint probability map by multiplication.

    :param prob: Marginal probability map of shape [n_batch, *vol_shape, n_atlas, n_class].
    :param mode: 'np' or 'tf'
    :return: A tensor of shape [n_batch, *vol_shape, n_class], representing the joint probability map.
    """
    assert mode in ['tf', 'np'], "The mode must be either 'tf' or 'np'!"
    # eps = kwargs.pop('eps', )
    if mode == 'tf':
        return tf.exp(tf.reduce_sum(tf.log(prob), axis=-2), name='joint_prob')
    if mode == 'np':
        return np.exp(np.sum(np.log(prob), axis=-2))


def get_normalized_prob(prob, mode='tf', **kwargs):
    """
    Compute normalized probability map given the input unnormalized probabilities.

    :param prob: The unnormalized probabilities of shape [n_batch, *vol_shape, n_class].
    :param mode: 'np' or 'tf'.
    :return: A tensor of shape [n_batch, *vol_shape, n_class], representing the probabilities of each class.
    """
    eps = kwargs.pop('eps', math.exp(-3**2/2))
    assert mode in ['tf', 'np'], "The mode must be either 'tf' or 'np'!"
    if mode == 'tf':
        prob = tf.clip_by_value(prob, eps, 1-eps)
        return tf.divide(prob, tf.reduce_sum(prob, axis=-1, keepdims=True), name='normalized_prob')
    elif mode == 'np':
        prob = np.clip(prob, eps, 1-eps)
        return np.divide(prob, np.sum(prob, axis=-1, keepdims=True))
    else:
        raise NotImplementedError


def get_prob_from_label(label, sigma=1., **kwargs):
    """
    Produce probability maps from one-hot labels.

    :param label: One-hot label of shape [n_batch, *vol_shape, n_class]
    :param sigma: The isotropic standard deviation of the Gaussian filter.
    :return: Probability map of shape [n_batch, *vol_shape, n_class]
    """
    eps = kwargs.pop('eps', math.exp(-3**2/2))
    mode = kwargs.pop('mode', 'tf')
    if mode == 'tf':
        with tf.name_scope('get_prob_from_label'):
            blur = separable_filter2d(label, gauss_kernel1d(sigma))
            prob = get_normalized_prob(blur, eps=eps)
    elif mode == 'np':
        blur = separable_filter2d(label, gauss_kernel1d(sigma), mode='np')
        prob = get_normalized_prob(blur, mode='np', eps=eps)
    else:
        raise NotImplementedError
    return prob


def get_atlases_prob_from_label(atlases_label, sigma=1., **kwargs):
    """
    Produce probabilistic atlases using isotropic Gaussian filters.

    :param atlas_labels: The atlas labels of shape [n_batch, *vol_shape, n_atlas, n_class].
    :param sigma: The isotropic standard deviation of the Gaussian filter.
    :param mode: 'tf' or 'np'
    :return: The probabilistic atlases of shape [n_batch, *vol_shape, n_atlas, n_class].
    """
    mode = kwargs.pop('mode', 'tf')
    eps = kwargs.pop('eps', math.exp(-3**2/2))

    if mode == 'tf':
        with tf.name_scope('get_atlases_prob'):
            n_atlas = atlases_label.get_shape().as_list()[-2]
            atlases_prob = tf.stack([get_prob_from_label(atlases_label[..., i, :], sigma, eps=eps)
                                     for i in range(n_atlas)], axis=-2, name='atlases_prob')

    elif mode == 'np':
        n_atlas = atlases_label.shape[-2]
        atlases_prob = np.stack([get_prob_from_label(atlases_label[..., i, :], sigma, eps=eps, mode=mode)
                                 for i in range(n_atlas)], axis=-2)

    else:
        raise ValueError("Unknown mode: %s" % mode)

    return atlases_prob


def compute_mask_from_prob(prob, **kwargs):
    """
    Compute the boundary mask from probability maps.

    :param prob: probability map of shape [n_batch, *vol_shape, n_class]
    :param kwargs:
    :return: a mask of shape [n_batch, *vol_shape, n_class]
    """
    mode = kwargs.pop('mode', 'tf')
    eps = kwargs.pop('eps', 1e-3)
    if mode == 'tf':
        with tf.name_scope('compute_mask_from_prob'):
            gradnorm = compute_gradnorm_from_volume(prob)
            # mask = tf.reduce_any(tf.greater(gradnorm, 0), axis=-1, name='mask')
            mask = tf.greater(gradnorm, eps)
            mask = tf.pad(mask[:, 3:-3, 3:-3, :], paddings=[[0, 0], [3, 3], [3, 3], [0, 0]])
            return tf.cast(mask, tf.float32)
    elif mode == 'np':
        gradnorm = compute_gradnorm_from_volume(prob, mode='np')
        mask = np.greater(gradnorm, eps)
        mask = np.pad(mask[:, 3:-3, 3:-3, :], pad_width=[[0, 0], [3, 3], [3, 3], [0, 0]], mode='constant')
        return mask.astype(np.float32)
    else:
        raise NotImplementedError


def compute_gradnorm_from_volume(vol, **kwargs):
    """
    Compute Euclidean norm of gradients from a volume.

    :param vol: a volume tensor of shape [n_batch, *vol_shape, channels]
    :return: a tensor of shape [n_batch, *vol_shape, channels]
    """
    mode = kwargs.pop('mode', 'tf')

    def gradient_dx(fv):
        return (fv[:, 2:, 1:-1] - fv[:, :-2, 1:-1]) / 2

    def gradient_dy(fv):
        return (fv[:, 1:-1, 2:] - fv[:, 1:-1, :-2]) / 2

    if mode == 'tf':
        with tf.name_scope('compute_gradient_from_volume'):
            channels = vol.get_shape().as_list()[-1]
            vol = tf.pad(vol, paddings=[[0, 0], [1, 1], [1, 1], [0, 0]], mode='SYMMETRIC')
            def gradient_pxyz(Ixyz, fn):
                return tf.stack([fn(Ixyz[..., i]) for i in range(channels)], axis=-1)
            dIdx = gradient_pxyz(vol, gradient_dx)  # [n_batch, *vol_shape, n_class]
            dIdy = gradient_pxyz(vol, gradient_dy)

            return tf.norm(tf.stack([dIdx, dIdy]), axis=0)

    elif mode == 'np':
        channels = vol.shape[-1]
        vol = np.pad(vol, pad_width=[[0, 0], [1, 1], [1, 1], [0, 0]], mode='edge')
        def gradient_pxyz(Ixyz, fn):
            return np.stack([fn(Ixyz[..., i]) for i in range(channels)], axis=-1)
        dIdx = gradient_pxyz(vol, gradient_dx)  # [n_batch, *vol_shape, n_class]
        dIdy = gradient_pxyz(vol, gradient_dy)

        return np.linalg.norm(np.stack([dIdx, dIdy]), axis=0)

    else:
        raise NotImplementedError


def gauss_kernel1d(sigma):
    if sigma == 0:
        return 0
    else:
        tail = int(sigma*3)
        k = np.exp([-0.5 * x ** 2 / sigma ** 2 for x in range(-tail, tail + 1)])
        return k / np.sum(k)


def separable_filter2d(vol, kernel, mode='tf'):
    """
    3D convolution using separable filter along each axis

    :param vol: of shape [batch, nx, ny, channels]
    :param kernel: of shape [k]
    :return: of shape [batch, nx, ny, channels]
    """
    if np.all(kernel == 0):
        return vol
    if mode == 'tf':
        kernel = tf.constant(kernel, dtype=tf.float32)
        channels = vol.get_shape().as_list()[-1]
        strides = [1, 1, 1, 1]
        return tf.concat([tf.nn.conv2d(tf.nn.conv2d(
            vol[..., i, None],
            tf.reshape(kernel, [-1, 1, 1, 1]), strides, "SAME"),
            tf.reshape(kernel, [1, -1, 1, 1]), strides, "SAME") for i in range(channels)], axis=-1)
    elif mode == 'np':
        return signal.convolve(signal.convolve(vol, np.reshape(kernel, [1, -1, 1, 1]), 'same'),
                               np.reshape(kernel, [1, 1, -1, 1]), 'same')


def gaussian_pdf(value, loc, scale):
    """
    Compute the Gaussian pdf of the given value with the mean and standard deviation.

    :param value: The value where to derive the pdf.
    :param loc: The means of Gaussian.
    :param scale: The standard deviations of Gaussian.
    :return: The pixel-wise Gaussian pdf.
    """
    with tf.name_scope('gaussian_pdf'):
        var = scale ** 2
        prob = tf.multiply(1 / tf.sqrt(2 * math.pi * var), tf.exp(- ((value - loc) ** 2) / (2 * var)),
                           name='gaussian_pdf')
        return prob


def gaussian_pdf_numpy(value, loc, scale):
    var = scale ** 2
    prob = 1 / np.sqrt(2 * math.pi * var) * np.exp(- ((value - loc) ** 2) / (2 * var))
    return prob


def reconstruct_grid_volume(grid, n_block=(4, 4, 4)):
    """
    Reconstruct the warped grid volume with its blocks.

    :param grid: A dictionary of grid blocks, with the key (i, j, k) referring to the block indices, the values are of
        shape [n_batch, nx_block, ny_block, nz_block, 3].
    :param n_block: The number of blocks along each axis.
    :return: The reconstructed grid volume of shape [n_batch, *vol_shape, 3].
    """
    with tf.name_scope('reconstruct_grid_volume'):
        return tf.concat([tf.concat([tf.concat([grid[(i, j, k)] for k in range(n_block[2])],
                                               axis=3) for j in range(n_block[1])],
                                    axis=2) for i in range(n_block[0])],
                         axis=1, name='reconstructed_grid_volume')


def get_reference_grid(grid_size):
    """
    Get coordinate matrices from coordinate vectors, deprecated for meshgrid for acceleration.

    :param grid_size: The size of the mesh grid.
    :return: A tensor of shape [*vol_shape, 3], with each element representing the three coordinates of the grids along
        one of the three axes.
    """
    with tf.name_scope('get_reference_grid'):
        return tf.to_float(tf.stack(tf.meshgrid([i for i in range(grid_size[0])],
                                                [j for j in range(grid_size[1])],
                                                [k for k in range(grid_size[2])],
                                                indexing='ij'),
                                    axis=3))


def get_reference_grid_numpy(grid_size):
    """
    Get coordinate matrices from grid size.

    :param grid_size: The size of the mesh grid.
    :return: An array of shape [*vol_shape, 3], representing the coordinates.
    """
    return np.stack(np.meshgrid([i for i in range(grid_size[0])],
                                [j for j in range(grid_size[1])],
                                [k for k in range(grid_size[2])],
                                indexing='ij'), axis=3)


def get_reference_grid_by_boundary(begins, ends):
    """
    Get coordinate matrices from coordinate vectors.

    :param begins: The begins of the mesh grid.
    :param ends: The ends of the mesh grid.
    :return: A tensor of shape [*vol_shape, 3].
    """
    with tf.name_scope('get_reference_grid_by_boundary'):
        return tf.to_float(tf.stack(tf.meshgrid(
            [i for i in range(begins[0], ends[0])],
            [j for j in range(begins[1], ends[1])],
            [k for k in range(begins[2], ends[2])],
            indexing='ij'), axis=3))


#######################################################################
# Helper functions.
#######################################################################

def remove_duplicates(lst: list):
    return sorted(set(lst), key=lst.index)


def config_logging(filename):
    # Create a custom logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Create handlers
    c_handler = logging.StreamHandler()
    f_handler = logging.FileHandler(filename)
    c_handler.setLevel(logging.INFO)
    f_handler.setLevel(logging.INFO)

    # Create formatters and add it to handlers
    c_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    f_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    c_handler.setFormatter(c_format)
    f_handler.setFormatter(f_format)

    # Add handlers to the logger
    logger.addHandler(c_handler)
    logger.addHandler(f_handler)

    return logger


def split_path_into_folders(path):
    """
    Split a given path into its folders.

    :param path: The given pathname.
    :return: A list containing folders.
    """
    folders = []
    while 1:
        path, folder = os.path.split(path)

        if folder != "":
            folders.append(folder)
        else:
            if path != "":
                folders.append(path)
            break

    folders.reverse()
    return folders



def strsort(alist):
    def atoi(text):
        return int(text) if text.isdigit() else text

    def natural_keys(text):
        """
        alist.sort(key=natural_keys) sorts in human order
        """
        return [atoi(c) for c in re.split('(\d+)', text)]

    alist.sort(key=natural_keys)
    return alist


def nCr(n, r):
    x = (fact(n) / (fact(r) * fact(n - r)))
    return x


# Returns factorial of n
def fact(n):
    res = 1
    for i in range(2, n + 1):
        res = res * i
    return res
