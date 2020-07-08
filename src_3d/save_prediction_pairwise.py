# -*- coding: utf-8 -*-
"""
Save atlases propagation results using registration with dense displacement fields predicted from networks.

@author: Xinzhe Luo

@version: 0.1
"""

from __future__ import print_function, division, absolute_import, unicode_literals
from core import model_ddf_mvmm_label_base as model
from core import image_dataset as image_utils
from core import utils, losses
# import nibabel as nib
import numpy as np
import os
import logging
import tensorflow as tf
import pandas as pd
import argparse
from datetime import datetime

t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='Start atlas propagation on test dataset!')
parser.add_argument('--time', default=t, type=str,
                    help='The current time to save test predictions.')
parser.add_argument('--space', type=str, default='commonspace2',
                    choices=['commonspace1', 'commonspace2'],
                    help='The commonspace type for test data.')
parser.add_argument('--spacing', default='2mm', type=str, choices=['1mm', '2mm'],
                    help='The spatial spacing of the network inputs and the dense displacement fields.')
parser.add_argument('--dropout', default=0, type=float, help='The dropout probability for network prediction.')
parser.add_argument('--dropout_type', default='regular', type=str, choices=['regular', 'spatial'],
                    help='dropout type')
parser.add_argument('--model_path', type=str, default=None,
                    help='The model path to restore the network parameters for network prediction.')
parser.add_argument('--latest_filename', default='best_checkpoint', type=str,
                    help='latest filename to restore the model')
parser.add_argument('--trial', default=0, type=int, help='which trial to load the model')
parser.add_argument('--cuda_device', default=0, type=int,
                    help='The cuda device for network prediction.')
parser.add_argument('--atlas_search_path', default='../../../dataset/training_mr_20_commonspace2/*.nii.gz', type=str,
                    help='The search pattern to find all training atlas images, labels and probabilities.')
parser.add_argument('--atlas_modality', default='mr', choices=['mr', 'ct'],
                    help="the modality of atlas image, either 'mr' or 'ct'")
parser.add_argument('--a_min', default=None, type=float, help='min value for intensity clipping')
parser.add_argument('--a_max', default=None, type=float, help='max value for intensity clipping')
parser.add_argument('--image_suffix', default='image.nii.gz', type=str,
                    help='suffix pattern for the images')
parser.add_argument('--label_suffix', default='label.nii.gz', type=str,
                    help='suffix pattern for the labels')
parser.add_argument('--weight_suffix', default=None, type=None,
                    help='suffix pattern for the weights')
parser.add_argument('--crop_patch', default=True, type=bool,
                    help='whether to crop patches of the test data')
parser.add_argument('--patch_center', default=None, nargs='+',
                    help='The customized patch center, default is None.')
parser.add_argument('--patch_size', default=(80, 80, 80), type=int, nargs='+',
                    help='The size of the cropped patch.')
parser.add_argument('--original_size', default=(112, 96, 112), type=int, nargs=3,
                    help='original size of the saved image')
parser.add_argument('--num_blocks', default=(1, 1, 1), type=int, nargs='+',
                    help='The number of blocks of input along each axis, default is (1, 1, 1).')
parser.add_argument('--method', default='unet',
                    choices=['ddf_label', 'ddf_label_v0', 'unet'], type=str,
                    help='the method of network to infer the dense displacement fields')
parser.add_argument('--num_down_blocks', default=4, type=int,
                    help='the number of downside convolution blocks of the network')
parser.add_argument('--ddf_levels', default=None, type=int, nargs='*',
                    help='the network levels where to extract dense displacement fields')
parser.add_argument('--features_root', default=32, type=int,
                    help='number of features of the first convolution layer')
parser.add_argument('--normalizer', default=None, type=str,
                    choices=['batch', 'group', 'layer', 'instance', 'batch_instance'],
                    help='type of network normalization method')
parser.add_argument('--diffeomorphism', default=False, action='store_true',
                    help='whether to use diffeomorphic transformations')
parser.add_argument('--int_steps', default=4, type=int,
                    help='number of integration steps on the velocity fields')
parser.add_argument('--cost_function', default='label_consistency',
                    choices=(['MvMM_negative_log-likelihood', 'label_consistency', 'multi_scale_label_consistency',
                              'dice', 'multi_scale_dice', 'cross_entropy', 'SSD']),
                    help='the type of cost function for network optimization')
parser.add_argument('--reg_stage', default='single', type=str, choices=['single', 'multi'],
                    help="The registration stage, either 'single' or 'multi'.")
parser.add_argument('--test_input_size', default=(112, 96, 112), type=int, nargs='+',
                    help='The test input size.')
parser.add_argument('--save_ddf', default=False, action='store_true',
                    help='whether to save displacement field into nifty files')
# parser.add_argument('--save_path', default='./', type=str,
#                     help="Path where to save the test results.")
args = parser.parse_args()

# determine the prediction/metrics save path and the data search path
if args.space == 'commonspace1':
    save_path = os.path.join(args.model_path, 'test_predictions_commonspace1_%s' % args.spacing)
    target_search_path = '../../../dataset/test_mr_40_commonspace1/*.nii.gz'
    metrics_path = os.path.join(args.model_path, 'metrics_test_pairwise_commonspace1_%s.xlsx' % args.spacing)
    scale_model = 1
elif args.space == 'commonspace2':
    save_path = os.path.join(args.model_path, 'test_predictions_commonspace2_%s' % args.spacing)
    target_search_path = '../../../dataset/test_mr_40_commonspace2/*.nii.gz'
    metrics_path = os.path.join(args.model_path, 'metrics_test_pairwise_commonspace2_%s.xlsx' % args.spacing)
    scale_model = 0
else:
    raise Exception("The space must be either 'commonspace1' or 'commonspace2'!")

if __name__ == '__main__':
    # set cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    # set working directory
    print("Current working directory: %s" % os.getcwd())
    os.chdir('../')
    print("Working directory changed to: %s" % os.path.abspath(os.getcwd()))

    if not os.path.exists(save_path):
        logging.info("Allocating '{:}'".format(save_path))
        os.makedirs(save_path)

    if 'model_trained' in args.model_path and 'trial' in args.model_path:
        model_path = args.model_path
    else:
        model_dir = os.path.join(args.model_path, 'trial_%s' % args.trial, 'model_trained')
        ckpt = tf.train.get_checkpoint_state(model_dir, latest_filename=args.latest_filename)
        model_path = ckpt.model_checkpoint_path

    test_model_data_provider = image_utils.ImageDataProvider(
        target_search_path='../../../dataset/test_mr_40_commonspace2/*.nii.gz',
        atlas_search_path=args.atlas_search_path,
        atlas_modality=args.atlas_modality,
        a_min=args.a_min,
        a_max=args.a_max,
        image_suffix=args.image_suffix,
        label_suffix=args.label_suffix,
        train_phase=False,
        n_atlas=1,
        crop_patch=args.crop_patch,
        patch_center=args.patch_center,
        patch_size=args.patch_size,
        crop_roi=False,
        image_normalization=True,
        channels=1,
        n_class=8,
        label_intensity=(0, 205, 420, 500, 550, 600, 820, 850),
        scale=0,
        num_blocks=args.num_blocks,
        image_name_index_begin=-38,
        stage=args.reg_stage)

    test_data_provider = image_utils.ImageDataProvider(target_search_path=target_search_path,
                                                       atlas_search_path=args.atlas_search_path,
                                                       atlas_modality=args.atlas_modality,
                                                       a_min=args.a_min,
                                                       a_max=args.a_max,
                                                       image_suffix=args.image_suffix,
                                                       label_suffix=args.label_suffix,
                                                       weight_suffix=args.weight_suffix,
                                                       train_phase=False,
                                                       n_atlas=1,
                                                       crop_patch=False,
                                                       # patch_center=[i*2**scale_model
                                                       #               for i in args.patch_center]
                                                       #              if args.patch_center else None,
                                                       # patch_size=[i*2**scale_model
                                                       #             for i in args.patch_size],
                                                       crop_roi=False,
                                                       image_normalization=False,
                                                       channels=1,
                                                       n_class=8,
                                                       label_intensity=(0, 205, 420, 500, 550, 600, 820, 850),
                                                       scale=0,
                                                       num_blocks=args.num_blocks,
                                                       stage=args.reg_stage,
                                                       image_name_index_begin=-38)

    logging.info("Number of target-atlas pairs: %s" % len(test_data_provider))

    with tf.Graph().as_default():
        net = model.NetForPrediction(n_blocks=args.num_blocks,
                                     test_input_size=args.test_input_size,
                                     input_scale=scale_model,
                                     input_size=args.patch_size,
                                     channels=1,
                                     n_class=2,
                                     test_n_class=8,
                                     n_atlas=1,
                                     n_subtypes=(2, 1),
                                     method=args.method,
                                     features_root=args.features_root,
                                     normalizer=args.normalizer,
                                     num_down_blocks=args.num_down_blocks,
                                     dropout_type=args.dropout_type,
                                     ddf_levels=args.ddf_levels,
                                     diffeomorphism=args.diffeomorphism,
                                     int_steps=args.int_steps,
                                     cost_kwargs={'cost_name': args.cost_function,
                                                  'regularizer': [None, 'bending_energy'],
                                                  'regularization_coefficient': [0., 1.]})

        # add number of negative Jacobians
        BendingEnergy = losses.LocalDisplacementEnergy('bending')
        jacobian_det = BendingEnergy.compute_jacobian_determinant(net.ddf)
        num_neg_jacob = tf.math.count_nonzero(tf.less_equal(jacobian_det, 0), dtype=tf.float32,
                                              name='negative_jacobians_number')
        setattr(net, 'num_neg_jacob', num_neg_jacob)

        # remove duplication of names
        frame_index = utils.remove_duplicates([os.path.split(pair_names[0])[-1]
                                                  for pair_names in test_data_provider.target_atlas_image_names])
        frame_columns = utils.remove_duplicates([os.path.split(pair_names[1][0])[-1][-39:]
                                                    for pair_names in test_data_provider.target_atlas_image_names])

        # list the metrics that need saving
        metrics_to_save = {'Dice': np.empty([len(frame_index), len(frame_columns)]),
                        'Jaccard': np.empty([len(frame_index), len(frame_columns)]),
                        'Myocardial Dice': np.empty([len(frame_index), len(frame_columns)]),
                        'LA Dice': np.empty([len(frame_index), len(frame_columns)]),
                        'LV Dice': np.empty([len(frame_index), len(frame_columns)]),
                        'RA Dice': np.empty([len(frame_index), len(frame_columns)]),
                        'RV Dice': np.empty([len(frame_index), len(frame_columns)]),
                        'AO Dice': np.empty([len(frame_index), len(frame_columns)]),
                        'PA Dice': np.empty([len(frame_index), len(frame_columns)]),
                        '# Negative Jacobians': np.empty([len(frame_index), len(frame_columns)]),
                           }

        with tf.Session(config=config) as sess:
            # Initialize variables
            sess.run(tf.global_variables_initializer())

            # Restore model parameters from previously saved model
            net.restore(sess, model_path, var_list=net.variables_to_restore)

            for idx, name in enumerate(test_data_provider.target_atlas_image_names):
                target_name = os.path.split(test_data_provider.target_atlas_image_names[idx][0])[-1]
                atlas_names = '-*-'.join([os.path.split(atlas_name)[-1] for atlas_name in test_data_provider.target_atlas_image_names[idx][1]])
                if args.space == 'commonspace1':
                    assert os.path.split(
                        test_model_data_provider.target_atlas_image_names[idx][0])[-1] == target_name.replace(
                        'commonspace1', 'commonspace2')
                    assert '-*-'.join(
                        [os.path.split(atlas_name)[-1]
                         for atlas_name in test_model_data_provider.target_atlas_image_names[idx][1]]) == atlas_names.replace(
                        'commonspace1', 'commonspace2')
                elif args.space == 'commonspace2':
                    assert os.path.split(test_model_data_provider.target_atlas_image_names[idx][0])[-1] == target_name
                    assert '-*-'.join(
                        [os.path.split(atlas_name)[-1]
                         for atlas_name in test_model_data_provider.target_atlas_image_names[idx][1]]) == atlas_names

                logging.info("Fixed image: Target {:}, "
                             "Moving image: Atlas {:}".format(target_name, atlas_names))

                # load data for network input
                model_data = test_model_data_provider[idx]
                # print(model_data['atlases_label'].shape, model_data['atlases_label'].dtype)

                # load data for label propagation and result evaluation
                test_data = test_data_provider[idx]

                # perform atlas transformation
                warped_atlas_image, warped_atlas_label, warped_atlas_weight,\
                    ddf, metrics = net.predict_scale(sess, model_data, test_data, args.dropout)
                # save metrics for the current target-atlas pair
                for k, v in metrics_to_save.items():
                    v[idx // len(frame_columns), idx % len(frame_columns)] = metrics[k]

                # save output into Nifty files
                # utils.save_prediction_nii(warped_atlas_image.squeeze(0), save_path, test_data_provider,
                #                              data_type='image', name_index=idx,
                #                              affine=test_data['target_affine'], header=test_data['target_header'],
                #                              save_suffix=args.image_suffix, stage=args.reg_stage,
                #                              # original_size=args.original_size
                #                              )

                utils.save_prediction_nii(warped_atlas_label.squeeze(0), save_path, test_data_provider,
                                          data_type='label', name_index=idx,
                                          affine=test_data['target_affine'], header=test_data['target_header'],
                                          save_suffix=args.label_suffix, stage=args.reg_stage,
                                          # original_size=args.original_size
                                          )

                if args.weight_suffix:
                    utils.save_prediction_nii(warped_atlas_weight.squeeze(0), save_path, test_data_provider,
                                                 data_type='image', name_index=idx,
                                                 affine=test_data['target_affine'], header=test_data['target_header'],
                                                 save_suffix=args.weight_suffix, save_dtype=np.float32, squeeze_channel=False,
                                                 stage=args.reg_stage,
                                                 # original_size=args.original_size
                                                 )

                if args.save_ddf:
                    utils.save_prediction_nii(ddf.squeeze((0, -2)), save_path, test_data_provider,
                                                 data_type='vector_fields', name_index=idx,
                                                 affine=test_data['target_affine'], header=test_data['target_header'],
                                                 stage=args.reg_stage, original_size=args.original_size)

    # save metrics into DataFrames
    metrics_DataFrames = {}
    for k, v in metrics_to_save.items():
        metrics_DataFrames[k] = pd.DataFrame(v, index=frame_index, columns=frame_columns, dtype=np.float32)

    # save metrics into excel files
    with pd.ExcelWriter(metrics_path) as writer:
        for k, v in metrics_DataFrames.items():
            v.to_excel(writer, sheet_name=k)
