# -*- coding: utf-8 -*-
"""
Save multi-atlas label fusion results for the test data from propagated atlases.

@author: Xinzhe Luo

@version: 0.1
"""

from __future__ import print_function, division, absolute_import, unicode_literals
# from core import model_ddf_mvmm_label_base as model
from core import image_dataset as image_utils
from core import utils
# import nibabel as nib
import numpy as np
# import tensorflow as tf
import os
import logging
import pandas as pd
import argparse
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
# config = tf.ConfigProto(device_count={'GPU': 0})  # cpu only
t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

parser = argparse.ArgumentParser(description="Start label fusion on propagated atlases!")
parser.add_argument('--model_path', type=str, required=True, 
                    help='model path where to load the pairwise label propagation results')
parser.add_argument('--method', default='multiplication', type=str,
                    choices=['multiply_mask', 'multiply_ncc', 'majority_voting'],
                    help='method to perform multi-atlas label fusion')
parser.add_argument('--atlas_search_path',
                    default='./2019-11-14_12-15-22_ddf_mvmm_label_mr_mr_2mm/test_predictions_commonspace2_2mm/*.nii.gz',
                    type=str, help='search pattern to find all atlas data')
parser.add_argument('--target_search_path',
                    default='../../../dataset/test_mr_40_commonspace2/*.nii.gz', type=str,
                    help='search pattern to find all target data')
parser.add_argument('--target_modality', default='mr', choices=["mr", "ct"],
                    help="the modality of target images, either 'mr' or 'ct'")
parser.add_argument('--atlas_modality', default='mr', choices=['mr', 'ct'],
                    help="the modality of atlas image, either 'mr' or 'ct'")
parser.add_argument('--image_suffix', default='image.nii.gz', type=str,
                    help='suffix pattern for image loading')
parser.add_argument('--label_suffix', default='label.nii.gz', type=str,
                    help='suffix pattern for label loading')
parser.add_argument('--weight_suffix', default=None, type=str,
                    help='suffix pattern for weight loading')
parser.add_argument('--channels', default=1, type=int, help='number of image channels')
parser.add_argument('--n_class', default=8, type=int, help='number of label classes, including the background')
parser.add_argument('--label_intensity', default=(0, 205, 420, 500, 550, 600, 820, 850), type=int, nargs='+',
                    help='list of intensities of the training ground truths')
parser.add_argument('--num_targets', default=40, type=int, help='number of targets to be fused')
parser.add_argument('--num_atlases', default=20, type=int, help='number of atlases to be fused')
parser.add_argument('--crop_patch', default=True, action='store_false',
                    help='whether patches of a certain size need to be cropped for training')
parser.add_argument('--patch_size', default=(80, 80, 80), type=int, nargs=3,
                    help='size of the training patches')
parser.add_argument('--original_size', default=(112, 96, 112), type=int, nargs=3,
                    help='original size of the saved image')
parser.add_argument('--crop_roi', default=False, action='store_true',
                    help='whether to crop ROI containing the whole foreground on training data')
parser.add_argument('--num_blocks', default=(1, 1, 1), type=int, nargs=3,
                    help='the number of blocks of input data along each axis')
parser.add_argument('--scale', default=0, type=int, help='scale of the processed data')
parser.add_argument('--sigma', default=1., type=float, help='scale of the Gaussian filter to produce prob labels')
parser.add_argument('--stage', default='multi', choices=['single', 'multi'],
                    help="the registration stage, either 'single' or 'multi', default 'multi'")
args = parser.parse_args()


if __name__ == '__main__':
    # change chief working directory
    os.chdir('../')
    logging.info("Chief working directory changed to: %s" % os.path.abspath(os.getcwd()))

    # tissue types
    tissue = {205: 'myocardium', 420: 'LA', 500: 'LV', 550: 'RA',
              600: 'RV', 820: 'ascending_aorta', 850: 'pulmonary_artery'}

    # where to save the fused labels
    save_path = os.path.join(args.model_path,
                             'label_fusions_commonspace2_atlases_%s_%s' % (args.num_atlases, args.method))
    # save_path = os.path.join(args.model_path,
    #                          'label_fusions_commonspace2_%s_atlases_%s_%s' % (tissue[args.label_intensity[1]],
    #                                                                           args.num_atlases, args.method))

    if not os.path.exists(save_path):
        logging.info("Allocating '%s'" % save_path)
        os.makedirs(save_path)

    # where to save the metrics
    metrics_path = os.path.join(args.model_path,
                                'metrics_fusions_commonspace2_%s.xlsx' % args.method)

    # create data provider
    data_provider = image_utils.ImageDataProvider(target_modality=args.target_modality,
                                                  atlas_modality=args.atlas_modality,
                                                  target_search_path=args.target_search_path,
                                                  atlas_search_path=args.atlas_search_path,
                                                  image_suffix=args.image_suffix,
                                                  label_suffix=args.label_suffix,
                                                  weight_suffix=args.weight_suffix,
                                                  n_atlas=args.num_atlases,
                                                  crop_patch=args.crop_patch,
                                                  patch_size=args.patch_size,
                                                  crop_roi=args.crop_roi,
                                                  channels=args.channels,
                                                  n_class=args.n_class,
                                                  label_intensity=args.label_intensity,
                                                  num_blocks=args.num_blocks,
                                                  scale=args.scale,
                                                  stage=args.stage)

    logging.info("Number of target-atlas pairs: %s" % len(data_provider))
    target_atlas_image_names = data_provider.target_atlas_image_names

    # set indices and columns for metrics saving into excel files
    frame_indices = utils.remove_duplicates([os.path.basename(pair_names[0]).replace(args.image_suffix, '')
                                                for pair_names in target_atlas_image_names])
    frame_columns = utils.remove_duplicates(['&'.join([os.path.basename(atlas_name)[-39:].replace(args.image_suffix, '')
                                                          for atlas_name in pair_names[1]])
                                                for pair_names in target_atlas_image_names])
    # print(frame_columns)

    # set metrics to save
    metrics_to_save = {'Dice': np.empty([len(frame_indices), len(frame_columns)]),
                       'Jaccard': np.empty([len(frame_indices), len(frame_columns)]),
                       'Myocardial Dice': np.empty([len(frame_indices), len(frame_columns)]),
                       'LA Dice': np.empty([len(frame_indices), len(frame_columns)]),
                       'LV Dice': np.empty([len(frame_indices), len(frame_columns)]),
                       'RA Dice': np.empty([len(frame_indices), len(frame_columns)]),
                       'RV Dice': np.empty([len(frame_indices), len(frame_columns)]),
                       'AO Dice': np.empty([len(frame_indices), len(frame_columns)]),
                       'PA Dice': np.empty([len(frame_indices), len(frame_columns)]),
                       'Average Surface Distance': np.empty([len(frame_indices), len(frame_columns)]),
                       'Hausdorff Distance': np.empty([len(frame_indices), len(frame_columns)])}

    logging.info("Start label fusion using method: %s" % args.method)
    # initialize fusion model
    fusion_model = utils.MvMMExpectationMaximization(n_class=args.n_class, spacing_mm=(2, 2, 2), sigma=args.sigma)

    # label fusion
    for idx in range(min(len(data_provider), args.num_targets)):
        # get target-atlases pair names
        target_name = os.path.basename(target_atlas_image_names[idx][0]).replace(args.image_suffix, '')
        atlases_name = '&'.join([os.path.basename(atlas_name)[-39:].replace(args.image_suffix, '')
                                 for atlas_name in target_atlas_image_names[idx][1]])

        assert target_name == frame_indices[idx // len(frame_columns)], "Target name: %s and frame index %s " \
                                                                        "should be equal!" % (target_name,
                                                                                              frame_indices[idx // len(frame_columns)])
        assert atlases_name == frame_columns[idx % len(frame_columns)], "Atlases name: %s and frame column %s " \
                                                                        "should be equal!" % (atlases_name,
                                                                                              frame_columns[idx % len(frame_columns)])

        logging.info("[Index]: %s; [Target]: %s; [Propagated atlases]: %s" % (idx, target_name, atlases_name))

        # load data
        data = data_provider[idx]

        # fuse labels
        fused_label, metrics = fusion_model.get_simple_fusion_result(warped_atlases_label=data['atlases_label'],
                                                                     target_labels=data['target_label'],
                                                                     method=args.method,
                                                                     warped_atlases_weight=data['atlases_weight'])
        # save metrics
        for k, v in metrics_to_save.items():
            v[idx // len(frame_columns), idx % len(frame_columns)] = metrics[k]

        # save the fused label
        utils.save_prediction_nii(fused_label.squeeze(0), save_path, data_provider, data_type='label',
                                     save_name=target_name + 'fusion.nii.gz',
                                     affine=data['target_affine'], header=data['target_header'],
                                     stage=args.stage, original_size=args.original_size)

    # convert metrics into DataFrames
    metrics_DataFrame = {}
    for k, v in metrics_to_save.items():
        metrics_DataFrame[k] = pd.DataFrame(v, index=frame_indices, columns=frame_columns, dtype=np.float32)

    # save metrics into excel files
    with pd.ExcelWriter(metrics_path) as writer:
        for k, v in metrics_DataFrame.items():
            v.to_excel(writer, sheet_name=k)
