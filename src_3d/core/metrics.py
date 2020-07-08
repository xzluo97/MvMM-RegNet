# -*- coding: utf-8 -*-
"""
Modules for computing metrics.

@author: Xinzhe Luo
"""

import numpy as np
import tensorflow as tf
from core import utils
from sklearn.metrics import roc_auc_score
from scipy import ndimage


##########################################################################
# Hyper classes for metrics construction.
##########################################################################
class OverlapMetrics(object):
    """
    Compute the Dice similarity coefficient between the ground truth and the prediction.

    """

    def __init__(self, n_class=1, eps=0.1, mode='tf'):

        self.n_class = n_class
        self.eps = eps
        self.mode = mode

        assert mode in ['tf', 'np'], "The mode must be either 'tf' or 'np'!"

    def averaged_foreground_dice(self, y_true, y_seg):
        """
        Assume the first class is the background.
        """
        if self.mode == 'tf':
            assert y_true.shape[1:] == y_seg.shape[1:], "The ground truth and prediction must be of equal shape! " \
                                                        "Ground truth shape: %s, " \
                                                        "prediction shape: %s" % (y_true.get_shape().as_list(),
                                                                                  y_seg.get_shape().as_list())

            assert y_seg.get_shape().as_list()[-1] == self.n_class, "The number of classes of the segmentation " \
                                                                    "should be equal to %s!" % self.n_class
        elif self.mode == 'np':
            assert y_true.shape == y_seg.shape, "The ground truth and prediction must be of equal shape! " \
                                                "Ground truth shape: %s, prediction shape: %s" % (y_true.shape,
                                                                                                  y_seg.shape)

            assert y_seg.shape[-1], "The number of classes of the segmentation should be equal to %s!" % self.n_class

        y_seg = utils.get_segmentation(y_seg, self.mode)
        dice = 0.
        if self.mode == 'tf':
            for i in range(1, self.n_class):
                top = 2 * tf.reduce_sum(y_true[..., i] * y_seg[..., i])
                bottom = tf.reduce_sum(y_true[..., i] + y_seg[..., i])
                dice += top / (tf.maximum(bottom, self.eps))

            return tf.divide(dice, tf.cast(self.n_class - 1, dtype=tf.float32), name='averaged_foreground_dice')

        elif self.mode == 'np':
            for i in range(1, self.n_class):
                top = 2 * np.sum(y_true[..., i] * y_seg[..., i])
                bottom = np.sum(y_true[..., i] + y_seg[..., i])
                dice += top / (np.maximum(bottom, self.eps))

            return np.divide(dice, self.n_class - 1)

    def class_specific_dice(self, y_true, y_seg, i):
        """
        Compute the class specific Dice.

        :param i: The i-th tissue class, default parameters: 0 for background; 1 for myocardium of the left ventricle;
            2 for left atrium; 3 for left ventricle; 4 for right atrium; 5 for right ventricle; 6 for ascending aorta;
            7 for pulmonary artery.
        """
        y_seg = utils.get_segmentation(y_seg, self.mode)

        if self.mode == 'tf':
            assert y_true.shape[1:] == y_seg.shape[1:], "The ground truth and prediction must be of equal shape! " \
                                                        "Ground truth shape: %s, " \
                                                        "prediction shape: %s" % (y_true.get_shape().as_list(),
                                                                                  y_seg.get_shape().as_list())

            top = 2 * tf.reduce_sum(y_true[..., i] * y_seg[..., i])
            bottom = tf.reduce_sum(y_true[..., i] + y_seg[..., i])
            dice = tf.divide(top, tf.maximum(bottom, self.eps), name='class%s_dice' % i)

        elif self.mode == 'np':
            assert y_true.shape == y_seg.shape, "The ground truth and prediction must be of equal shape! " \
                                                "Ground truth shape: %s, prediction shape: %s" % (y_true.shape,
                                                                                                  y_seg.shape)

            top = 2 * np.sum(y_true[..., i] * y_seg[..., i])
            bottom = np.sum(y_true[..., i] + y_seg[..., i])
            dice = np.divide(top, np.maximum(bottom, self.eps))

        return dice

    def averaged_foreground_jaccard(self, y_true, y_seg):
        """
                Assume the first class is the background.
                """
        if self.mode == 'tf':
            assert y_true.shape[1:] == y_seg.shape[1:], "The ground truth and prediction must be of equal shape! " \
                                                        "Ground truth shape: %s, " \
                                                        "prediction shape: %s" % (y_true.get_shape().as_list(),
                                                                                  y_seg.get_shape().as_list())

            assert y_seg.get_shape().as_list()[-1] == self.n_class, "The number of classes of the segmentation " \
                                                                    "should be equal to %s!" % self.n_class
        elif self.mode == 'np':
            assert y_true.shape == y_seg.shape, "The ground truth and prediction must be of equal shape! " \
                                                "Ground truth shape: %s, prediction shape: %s" % (y_true.shape,
                                                                                                  y_seg.shape)

            assert y_seg.shape[-1], "The number of classes of the segmentation should be equal to %s!" % self.n_class

        y_seg = utils.get_segmentation(y_seg, self.mode)
        jaccard = 0.
        if self.mode == 'tf':
            y_true = tf.cast(y_true, dtype=tf.bool)
            y_seg = tf.cast(y_seg, dtype=tf.bool)
            for i in range(1, self.n_class):
                top = tf.reduce_sum(tf.cast(tf.logical_and(y_true[..., i], y_seg[..., i]), tf.float32))
                bottom = tf.reduce_sum(tf.cast(tf.logical_or(y_true[..., i], y_seg[..., i]), tf.float32))
                jaccard += top / (tf.maximum(bottom, self.eps))

            return tf.divide(jaccard, tf.cast(self.n_class - 1, dtype=tf.float32),
                             name='averaged_foreground_jaccard')

        elif self.mode == 'np':
            y_true = y_true.astype(np.bool)
            y_seg = y_seg.astype(np.bool)
            for i in range(1, self.n_class):
                top = np.sum(np.logical_and(y_true[..., i], y_seg[..., i]).astype(np.float32))
                bottom = np.sum(np.logical_or(y_true[..., i], y_seg[..., i]).astype(np.float32))
                jaccard += top / (np.maximum(bottom, self.eps))

            return np.divide(jaccard, self.n_class - 1)


class SurfaceDistance(object):
    """
    Module computing surface distance based measures.
    Modified from https://github.com/deepmind/surface-distance.

    :param spacing_mm: 3-element list-like structure, indicating voxel spacings in x, y, z direction.
    """

    def __init__(self, spacing_mm=(1, 1, 1)):
        self.spacing_mm = spacing_mm

    def compute_surface_distances(self, mask_gt, mask_pred):
        """Compute closest distances from all surface points to the other surface.
        Finds all surface elements "surfels" in the ground truth mask `mask_gt` and
        the predicted mask `mask_pred`, computes their area in mm^2 and the distance
        to the closest point on the other surface. It returns two sorted lists of
        distances together with the corresponding surfel areas. If one of the masks
        is empty, the corresponding lists are empty and all distances in the other
        list are `inf`.
        Args:
          mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
          mask_pred: 3-dim Numpy array of type bool. The predicted mask.
          spacing_mm: 3-element list-like structure. Voxel spacing in x0, x1 and x2
              direction.
        Returns:
          A dict with:
          "distances_gt_to_pred": 1-dim numpy array of type float. The distances in mm
              from all ground truth surface elements to the predicted surface,
              sorted from smallest to largest.
          "distances_pred_to_gt": 1-dim numpy array of type float. The distances in mm
              from all predicted surface elements to the ground truth surface,
              sorted from smallest to largest.
          "surfel_areas_gt": 1-dim numpy array of type float. The area in mm^2 of
              the ground truth surface elements in the same order as
              distances_gt_to_pred
          "surfel_areas_pred": 1-dim numpy array of type float. The area in mm^2 of
              the predicted surface elements in the same order as
              distances_pred_to_gt
        """

        # compute the area for all 256 possible surface elements
        # (given a 2x2x2 neighbourhood) according to the spacing_mm
        neighbour_code_to_surface_area = np.zeros([256])
        for code in range(256):
            normals = np.array(neighbour_code_to_normals[code])
            sum_area = 0
            for normal_idx in range(normals.shape[0]):
                # normal vector
                n = np.zeros([3])
                n[0] = normals[normal_idx, 0] * self.spacing_mm[1] * self.spacing_mm[2]
                n[1] = normals[normal_idx, 1] * self.spacing_mm[0] * self.spacing_mm[2]
                n[2] = normals[normal_idx, 2] * self.spacing_mm[0] * self.spacing_mm[1]
                area = np.linalg.norm(n)
                sum_area += area
            neighbour_code_to_surface_area[code] = sum_area

        # compute the bounding box of the masks to trim
        # the volume to the smallest possible processing subvolume
        mask_gt = np.asarray(mask_gt, np.bool)
        mask_pred = np.asarray(mask_pred, np.bool)
        mask_all = mask_gt | mask_pred
        bbox_min = np.zeros(3, np.int64)
        bbox_max = np.zeros(3, np.int64)

        # max projection to the x0-axis
        proj_0 = np.max(np.max(mask_all, axis=2), axis=1)
        idx_nonzero_0 = np.nonzero(proj_0)[0]
        if len(idx_nonzero_0) == 0:  # pylint: disable=g-explicit-length-test
            return {"distances_gt_to_pred": np.array([]),
                    "distances_pred_to_gt": np.array([]),
                    "surfel_areas_gt": np.array([]),
                    "surfel_areas_pred": np.array([])}

        bbox_min[0] = np.min(idx_nonzero_0)
        bbox_max[0] = np.max(idx_nonzero_0)

        # max projection to the x1-axis
        proj_1 = np.max(np.max(mask_all, axis=2), axis=0)
        idx_nonzero_1 = np.nonzero(proj_1)[0]
        bbox_min[1] = np.min(idx_nonzero_1)
        bbox_max[1] = np.max(idx_nonzero_1)

        # max projection to the x2-axis
        proj_2 = np.max(np.max(mask_all, axis=1), axis=0)
        idx_nonzero_2 = np.nonzero(proj_2)[0]
        bbox_min[2] = np.min(idx_nonzero_2)
        bbox_max[2] = np.max(idx_nonzero_2)

        # crop the processing subvolume.
        # we need to zeropad the cropped region with 1 voxel at the lower,
        # the right and the back side. This is required to obtain the "full"
        # convolution result with the 2x2x2 kernel
        cropmask_gt = np.zeros((bbox_max - bbox_min) + 2, np.uint8)
        cropmask_pred = np.zeros((bbox_max - bbox_min) + 2, np.uint8)

        cropmask_gt[0:-1, 0:-1, 0:-1] = mask_gt[bbox_min[0]:bbox_max[0] + 1,
                                        bbox_min[1]:bbox_max[1] + 1,
                                        bbox_min[2]:bbox_max[2] + 1]

        cropmask_pred[0:-1, 0:-1, 0:-1] = mask_pred[bbox_min[0]:bbox_max[0] + 1,
                                          bbox_min[1]:bbox_max[1] + 1,
                                          bbox_min[2]:bbox_max[2] + 1]

        # compute the neighbour code (local binary pattern) for each voxel
        # the resultsing arrays are spacially shifted by minus half a voxel in each
        # axis.
        # i.e. the points are located at the corners of the original voxels
        kernel = np.array([[[128, 64],
                            [32, 16]],
                           [[8, 4],
                            [2, 1]]])
        neighbour_code_map_gt = ndimage.filters.correlate(
            cropmask_gt.astype(np.uint8), kernel, mode="constant", cval=0)
        neighbour_code_map_pred = ndimage.filters.correlate(
            cropmask_pred.astype(np.uint8), kernel, mode="constant", cval=0)

        # create masks with the surface voxels
        borders_gt = ((neighbour_code_map_gt != 0) & (neighbour_code_map_gt != 255))
        borders_pred = ((neighbour_code_map_pred != 0) &
                        (neighbour_code_map_pred != 255))

        # compute the distance transform (closest distance of each voxel to the
        # surface voxels)
        if borders_gt.any():
            distmap_gt = ndimage.morphology.distance_transform_edt(
                ~borders_gt, sampling=self.spacing_mm)
        else:
            distmap_gt = np.Inf * np.ones(borders_gt.shape)

        if borders_pred.any():
            distmap_pred = ndimage.morphology.distance_transform_edt(
                ~borders_pred, sampling=self.spacing_mm)
        else:
            distmap_pred = np.Inf * np.ones(borders_pred.shape)

        # compute the area of each surface element
        surface_area_map_gt = neighbour_code_to_surface_area[neighbour_code_map_gt]
        surface_area_map_pred = neighbour_code_to_surface_area[
            neighbour_code_map_pred]

        # create a list of all surface elements with distance and area
        distances_gt_to_pred = distmap_pred[borders_gt]
        distances_pred_to_gt = distmap_gt[borders_pred]
        surfel_areas_gt = surface_area_map_gt[borders_gt]
        surfel_areas_pred = surface_area_map_pred[borders_pred]

        # sort them by distance
        if distances_gt_to_pred.shape != (0,):
            sorted_surfels_gt = np.array(
                sorted(zip(distances_gt_to_pred, surfel_areas_gt)))
            distances_gt_to_pred = sorted_surfels_gt[:, 0]
            surfel_areas_gt = sorted_surfels_gt[:, 1]

        if distances_pred_to_gt.shape != (0,):
            sorted_surfels_pred = np.array(
                sorted(zip(distances_pred_to_gt, surfel_areas_pred)))
            distances_pred_to_gt = sorted_surfels_pred[:, 0]
            surfel_areas_pred = sorted_surfels_pred[:, 1]

        return {"distances_gt_to_pred": distances_gt_to_pred,
                "distances_pred_to_gt": distances_pred_to_gt,
                "surfel_areas_gt": surfel_areas_gt,
                "surfel_areas_pred": surfel_areas_pred}

    def compute_average_surface_distance(self, mask_gt, mask_pred):
        """Returns the average surface distance.
        Computes the average surface distances by correctly taking the area of each
        surface element into account. Call compute_surface_distances(...) before, to
        obtain the `surface_distances` dict.
        Args:
          mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
          mask_pred: 3-dim Numpy array of type bool. The predicted mask.
        Returns:
          A tuple with two float values: the average distance (in mm) from the
          ground truth surface to the predicted surface and the average distance from
          the predicted surface to the ground truth surface.
        """
        surface_distances = self.compute_surface_distances(mask_gt, mask_pred)
        distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        surfel_areas_gt = surface_distances["surfel_areas_gt"]
        surfel_areas_pred = surface_distances["surfel_areas_pred"]
        # average_distance_gt_to_pred = (
        #         np.sum(distances_gt_to_pred * surfel_areas_gt) / np.sum(surfel_areas_gt))
        # average_distance_pred_to_gt = (
        #         np.sum(distances_pred_to_gt * surfel_areas_pred) /
        #         np.sum(surfel_areas_pred))
        # return (average_distance_gt_to_pred, average_distance_pred_to_gt)
        return (np.sum(distances_gt_to_pred * surfel_areas_gt) +
                np.sum(distances_pred_to_gt * surfel_areas_pred)) / (np.sum(surfel_areas_gt) +
                                                                     np.sum(surfel_areas_pred))

    def compute_robust_hausdorff(self, mask_gt, mask_pred, percent=100):
        """Computes the robust Hausdorff distance.
        Computes the robust Hausdorff distance. "Robust", because it uses the
        `percent` percentile of the distances instead of the maximum distance. The
        percentage is computed by correctly taking the area of each surface element
        into account.
        Args:
          mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
          mask_pred: 3-dim Numpy array of type bool. The predicted mask.
          percent: a float value between 0 and 100.
        Returns:
          a float value. The robust Hausdorff distance in mm.
        """
        surface_distances = self.compute_surface_distances(mask_gt, mask_pred)
        distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        surfel_areas_gt = surface_distances["surfel_areas_gt"]
        surfel_areas_pred = surface_distances["surfel_areas_pred"]
        if len(distances_gt_to_pred) > 0:  # pylint: disable=g-explicit-length-test
            surfel_areas_cum_gt = np.cumsum(surfel_areas_gt) / np.sum(surfel_areas_gt)
            idx = np.searchsorted(surfel_areas_cum_gt, percent / 100.0)
            perc_distance_gt_to_pred = distances_gt_to_pred[
                min(idx, len(distances_gt_to_pred) - 1)]
        else:
            perc_distance_gt_to_pred = np.Inf

        if len(distances_pred_to_gt) > 0:  # pylint: disable=g-explicit-length-test
            surfel_areas_cum_pred = (np.cumsum(surfel_areas_pred) /
                                     np.sum(surfel_areas_pred))
            idx = np.searchsorted(surfel_areas_cum_pred, percent / 100.0)
            perc_distance_pred_to_gt = distances_pred_to_gt[
                min(idx, len(distances_pred_to_gt) - 1)]
        else:
            perc_distance_pred_to_gt = np.Inf

        return max(perc_distance_gt_to_pred, perc_distance_pred_to_gt)

    def compute_surface_overlap_at_tolerance(self, mask_gt, mask_pred, tolerance_mm):
        """Computes the overlap of the surfaces at a specified tolerance.
        Computes the overlap of the ground truth surface with the predicted surface
        and vice versa allowing a specified tolerance (maximum surface-to-surface
        distance that is regarded as overlapping). The overlapping fraction is
        computed by correctly taking the area of each surface element into account.
        Args:
          mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
          mask_pred: 3-dim Numpy array of type bool. The predicted mask.
          tolerance_mm: a float value. The tolerance in mm
        Returns:
          A tuple of two float values. The overlap fraction (0.0 - 1.0) of the ground
          truth surface with the predicted surface and vice versa.
        """
        surface_distances = self.compute_surface_distances(mask_gt, mask_pred)
        distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        surfel_areas_gt = surface_distances["surfel_areas_gt"]
        surfel_areas_pred = surface_distances["surfel_areas_pred"]
        rel_overlap_gt = (
                np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm]) /
                np.sum(surfel_areas_gt))
        rel_overlap_pred = (
                np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm]) /
                np.sum(surfel_areas_pred))
        return (rel_overlap_gt, rel_overlap_pred)

    def compute_surface_dice_at_tolerance(self, mask_gt, mask_pred, tolerance_mm):
        """Computes the _surface_ DICE coefficient at a specified tolerance.
        Computes the _surface_ DICE coefficient at a specified tolerance. Not to be
        confused with the standard _volumetric_ DICE coefficient. The surface DICE
        measaures the overlap of two surfaces instead of two volumes. A surface
        element is counted as overlapping (or touching), when the closest distance to
        the other surface is less or equal to the specified tolerance. The DICE
        coefficient is in the range between 0.0 (no overlap) to 1.0 (perfect overlap).
        Args:
          mask_gt: 3-dim Numpy array of type bool. The ground truth mask.
          mask_pred: 3-dim Numpy array of type bool. The predicted mask.
          tolerance_mm: a float value. The tolerance in mm
        Returns:
          A float value. The surface DICE coefficient (0.0 - 1.0).
        """
        surface_distances = self.compute_surface_distances(mask_gt, mask_pred)
        distances_gt_to_pred = surface_distances["distances_gt_to_pred"]
        distances_pred_to_gt = surface_distances["distances_pred_to_gt"]
        surfel_areas_gt = surface_distances["surfel_areas_gt"]
        surfel_areas_pred = surface_distances["surfel_areas_pred"]
        overlap_gt = np.sum(surfel_areas_gt[distances_gt_to_pred <= tolerance_mm])
        overlap_pred = np.sum(surfel_areas_pred[distances_pred_to_gt <= tolerance_mm])
        surface_dice = (overlap_gt + overlap_pred) / (
                np.sum(surfel_areas_gt) + np.sum(surfel_areas_pred))
        return surface_dice


############################################################################
# Low-level instantiation for metrics computing
############################################################################

def average_foreground_dice(predictions, labels):
    """
    Return the dice score based on dense predictions and labels.
    :param predictions: list of output predictions
    :param labels: list of ground truths
    """
    assert len(predictions) == len(labels), "Number of predictions and labels don't equal."
    n_class = labels[0].shape[-1]
    dice = np.array([])
    n = len(predictions)
    for i in range(n):
        pred = np.array(predictions[i])
        label = utils.crop_to_shape(np.array(labels[i]), pred.shape)
        Dice = OverlapMetrics(n_class, mode='np')
        dice = np.hstack((dice, Dice.averaged_foreground_dice(label, pred)))

    return dice


def myocardial_dice_score(predictions, labels):
    """
    Return the myocardial dice score between predictions and ground truths.

    :param predictions: list of output predictions
    :param labels: list of ground truths
    :return: a list of myocardial dice score
    """

    assert len(predictions) == len(labels), "Number of predictions and labels don't equal."

    n = len(predictions)
    dice = np.zeros([n])
    for i in range(n):
        pred = np.array(predictions[i])
        label = utils.crop_to_shape(np.array(labels[i]), pred.shape)
        dice[i] = OverlapMetrics(mode='np').class_specific_dice(label, pred, i=1)

    return dice


def average_foreground_jaccard(predictions, labels):
    """
    Return the averaged foreground Jaccard based on dense predictions and labels.

    :param predictions: list of output predictions
    :param labels: list of ground truths
    """
    n_class = labels[0].shape[-1]
    n = len(predictions)
    jaccard = np.empty([n])
    for i in range(n):
        pred = np.array(predictions[i])
        label = utils.crop_to_shape(np.array(labels[i]), pred.shape)
        Jaccard = OverlapMetrics(n_class, mode='np')
        jaccard[i] = Jaccard.averaged_foreground_jaccard(label, pred)

    return jaccard


def acc_rate(predictions, labels):
    """
    Return the error rate based on dense predictions and labels.
    :param predictions: list of output predictions
    :param labels: list of ground truths
    """
    assert len(predictions) == len(labels), "Number of predictions and labels don't equal."
    err = np.array([])
    n = len(predictions)
    for i in range(n):
        err = np.hstack((err, (100.0 * np.average(
            np.argmax(predictions[i], -1) == np.argmax(utils.crop_to_shape(labels[i], predictions[i].shape), -1)))))
    return err


def auc_score(predictions, labels):
    """
    Return the auc score based on dense predictions and labels.
    :param predictions: list of output predictions
    :param labels: list of ground truths
    """
    assert len(predictions) == len(labels), "Number of predictions and labels don't equal."
    auc = np.array([])
    n = len(predictions)
    n_class = np.shape(labels[0])[-1]
    for i in range(n):
        flat_score = np.reshape(predictions[i], [-1, n_class])
        flat_true = np.reshape(utils.crop_to_shape(labels[i], predictions[i].shape), [-1, n_class])
        auc = np.hstack((auc, roc_auc_score(flat_true, flat_score)))
    return auc


def average_surface_distance(predictions, labels, spacing_mm=(1, 1, 1)):
    """
    Return the average surface distances based on the predictions and labels.

    :param predictions: list of output predictions, of shape [1, *vol_shape, n_class]
    :param labels: list of ground truths, of shape [1, *vol_shape, n_class]
    :param spacing_mm: 3-element indicating voxel spacings in x, y, z direction.
    :return: An array of the ASD metric of each prediction.
    """
    assert len(predictions) == len(labels), "Number of predictions and labels don't equal."

    n = len(predictions)
    asd = np.empty([n])
    n_class = labels[0].shape[-1]
    SD = SurfaceDistance(spacing_mm)
    for i in range(n):
        class_asd = []
        mask_gt = labels[i].squeeze(0)
        mask_pred = utils.get_segmentation(predictions[i], mode='np').squeeze(0)
        for k in range(n_class):
            class_asd.append(SD.compute_average_surface_distance(mask_gt=mask_gt[..., k],
                                                                 mask_pred=mask_pred[..., k])
                             )

        asd[i] = np.mean(class_asd)

    return asd


def hausdorff_distance(predictions, labels, spacing_mm=(1, 1, 1), percent=100):
    """
    Return the Hausdorff distances based on the predictions and labels.

    :param predictions: list of output predictions
    :param labels: list of ground truths
    :param spacing_mm: 3-element indicating voxel spacings in x, y, z direction.
    :param percent: percentile of the distances instead of the maximum distance, a value between 0 and 100
    :return: An array of the ASD metric of each prediction.
    """
    assert len(predictions) == len(labels), "Number of predictions and labels don't equal."

    n = len(predictions)
    hd = np.empty([n])
    n_class = labels[0].shape[-1]
    SD = SurfaceDistance(spacing_mm)
    for i in range(n):
        class_hd = []
        mask_gt = labels[i].squeeze(0)
        mask_pred = utils.get_segmentation(predictions[i], mode='np').squeeze(0)
        for k in range(n_class):
            class_hd.append(SD.compute_robust_hausdorff(mask_gt=mask_gt[..., k],
                                                        mask_pred=mask_pred[..., k],
                                                        percent=percent)
                            )

        hd[i] = np.mean(class_hd)

    return hd


########################################################################
# neighbour_code_to_normals is a lookup table.
# For every binary neighbour code
# (2x2x2 neighbourhood = 8 neighbours = 8 bits = 256 codes)
# it contains the surface normals of the triangles (called "surfel" for
# "surface element" in the following). The length of the normal
# vector encodes the surfel area.
#
# created using the marching_cube algorithm
# see e.g. https://en.wikipedia.org/wiki/Marching_cubes
neighbour_code_to_normals = [
    [[0, 0, 0]],
    [[0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
    [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
    [[0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
    [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375], [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
    [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
    [[0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25], [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
    [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375], [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
    [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
    [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25], [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
    [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25], [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
    [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375], [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
    [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375], [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
    [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0], [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125], [0.125, 0.125, 0.125]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.125, 0.125, 0.125], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[-0.375, -0.375, 0.375], [0.25, -0.25, 0.0], [0.0, 0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.0, -0.5, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[0.375, -0.375, 0.375], [0.0, 0.25, 0.25], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.5, 0.0, -0.0], [0.25, -0.25, -0.25], [0.125, -0.125, -0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25], [0.0, 0.25, 0.25]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[0.125, 0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.25, 0.0, 0.25], [-0.375, -0.375, 0.375], [-0.25, 0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25], [0.25, 0.0, 0.25]],
    [[-0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25], [-0.125, 0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[0.125, 0.125, 0.125], [0.125, 0.125, 0.125], [0.25, 0.25, 0.25], [0.0, 0.0, 0.5]],
    [[-0.0, 0.0, 0.5], [0.0, 0.0, 0.5]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [-0.375, 0.375, 0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[-0.0, 0.0, 0.5], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [0.25, 0.0, -0.25]],
    [[0.5, 0.0, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[-0.25, 0.0, -0.25], [0.375, -0.375, -0.375], [0.0, 0.25, -0.25], [-0.125, 0.125, 0.125]],
    [[-0.25, 0.25, -0.25], [-0.25, 0.25, -0.25], [-0.125, 0.125, -0.125], [-0.125, 0.125, -0.125]],
    [[-0.0, 0.5, 0.0], [-0.25, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[0.375, -0.375, 0.375], [0.0, -0.25, -0.25], [-0.125, 0.125, -0.125], [0.25, 0.25, 0.0]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.0, 0.0, 0.5], [0.25, -0.25, 0.25], [0.125, -0.125, 0.125]],
    [[0.0, -0.25, 0.25], [0.0, -0.25, 0.25]],
    [[-0.125, -0.125, 0.125], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[0.125, 0.125, 0.125], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125]],
    [[-0.375, 0.375, -0.375], [-0.25, -0.25, 0.0], [-0.125, 0.125, -0.125], [-0.25, 0.0, 0.25]],
    [[0.0, 0.5, 0.0], [0.25, 0.25, -0.25], [-0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[0.125, 0.125, 0.125], [0.0, -0.5, 0.0], [-0.25, -0.25, -0.25], [-0.125, -0.125, -0.125]],
    [[-0.375, -0.375, -0.375], [-0.25, 0.0, 0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0], [0.125, -0.125, 0.125]],
    [[0.0, 0.5, 0.0], [0.0, -0.5, 0.0]],
    [[0.0, 0.5, 0.0], [0.125, -0.125, 0.125], [-0.25, 0.25, -0.25]],
    [[0.0, 0.5, 0.0], [-0.25, 0.25, 0.25], [0.125, -0.125, -0.125]],
    [[0.25, -0.25, 0.0], [-0.25, 0.25, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.0, 0.25, -0.25], [0.375, -0.375, -0.375], [-0.125, 0.125, 0.125], [0.25, 0.25, 0.0]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.25, 0.25, -0.25], [0.25, 0.25, -0.25], [0.125, 0.125, -0.125], [-0.125, -0.125, 0.125]],
    [[-0.0, 0.0, 0.5], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[-0.375, -0.375, 0.375], [-0.0, 0.25, 0.25], [0.125, 0.125, -0.125], [-0.25, -0.0, -0.25]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.0, -0.5, 0.0], [0.125, 0.125, -0.125], [0.25, 0.25, -0.25]],
    [[0.0, -0.25, 0.25], [0.0, 0.25, -0.25]],
    [[0.125, 0.125, 0.125], [0.125, -0.125, 0.125]],
    [[0.125, -0.125, 0.125]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25]],
    [[-0.5, 0.0, 0.0], [-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.125, 0.125, 0.125]],
    [[0.375, 0.375, 0.375], [0.0, 0.25, -0.25], [-0.125, -0.125, -0.125], [-0.25, 0.25, 0.0]],
    [[0.125, -0.125, -0.125], [0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.125, 0.125, 0.125], [0.375, 0.375, 0.375], [0.0, -0.25, 0.25], [-0.25, 0.0, 0.25]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125], [0.125, -0.125, -0.125]],
    [[-0.125, -0.125, -0.125], [-0.25, -0.25, -0.25], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, 0.0, -0.5], [0.25, 0.25, 0.25], [-0.125, -0.125, -0.125]],
    [[0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.5, 0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[-0.125, -0.125, 0.125], [0.125, -0.125, -0.125]],
    [[0.0, -0.25, -0.25], [0.0, 0.25, 0.25]],
    [[0.125, -0.125, -0.125]],
    [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
    [[-0.5, 0.0, 0.0], [-0.25, 0.25, 0.25], [-0.125, 0.125, 0.125]],
    [[0.5, 0.0, 0.0], [0.25, -0.25, 0.25], [-0.125, 0.125, -0.125]],
    [[0.25, -0.25, 0.0], [0.25, -0.25, 0.0]],
    [[0.5, 0.0, 0.0], [-0.25, -0.25, 0.25], [-0.125, -0.125, 0.125]],
    [[-0.25, 0.0, 0.25], [-0.25, 0.0, 0.25]],
    [[0.125, 0.125, 0.125], [-0.125, 0.125, 0.125]],
    [[-0.125, 0.125, 0.125]],
    [[0.5, 0.0, -0.0], [0.25, 0.25, 0.25], [0.125, 0.125, 0.125]],
    [[0.125, -0.125, 0.125], [-0.125, -0.125, 0.125]],
    [[-0.25, -0.0, -0.25], [0.25, 0.0, 0.25]],
    [[0.125, -0.125, 0.125]],
    [[-0.25, -0.25, 0.0], [0.25, 0.25, -0.0]],
    [[-0.125, -0.125, 0.125]],
    [[0.125, 0.125, 0.125]],
    [[0, 0, 0]]]
