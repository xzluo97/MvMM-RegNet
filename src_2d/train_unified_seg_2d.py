# -*- coding: utf-8 -*-
"""
Created on Fri Aug 16 16:24:00 2019

Unified multi-atlas segmentation using multivariate mixture model with 3-D dense displacement fields of resolution 2mm.

@author: Xinzhe Luo

@version: 0.1
"""

from __future__ import print_function, division, absolute_import, unicode_literals
from core import model_2d_ddf_mvmm_label_base as model
from core import image_2d_dataset as image_utils
from core import utils_2d
# import numpy as np
import argparse
import os
import logging
from datetime import datetime
import tensorflow as tf
# import math

t = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

parser = argparse.ArgumentParser("MvMM-RegNet for MSCMR dataset")

# overall setups
parser.add_argument('--cuda_device', default=0, type=int, choices=list(range(-1, 4)),
                    help='the cuda device to use for network training/validation')
parser.add_argument('--spacing', default='1.5mm',
                    help='the spatial spacing of the network inputs and the output vector fields')
parser.add_argument('--model_type', default='ddf_mvmm_label', choices=['ddf_mvmm', 'ddf_mvmm_label'],
                    help="the model type, 'ddf_mvmm' for unsupervised learning and 'ddf_mvmm_label' for "
                         "weakly-supervised learning")
parser.add_argument('--training_times', default=1, type=int, help='model training times')
parser.add_argument('--time', default=t, type=str,
                    help='the current time or the time when the model to restore was trained')
parser.add_argument('--target_modalities', default=('DE', ), choices=["C0", "DE", "T2"], nargs='+',
                    help="the modalities of target images, among 'C0', 'DE' and 'T2'")
parser.add_argument('--atlas_modalities', default=('C0', ), choices=['C0', 'DE', 'T2'], nargs='+',
                    help="the modalities of atlas images, among 'C0', 'DE' and 'T2'")
parser.add_argument('--a_min', default=None, type=float, help='min value for intensity clipping')
parser.add_argument('--a_max', default=None, type=float, help='max value for intensity clipping')
parser.add_argument('--d_baseline', default=10, type=float,
                    help='slices within at least d_baseline distance will be taken into account as image pairs')
parser.add_argument('--max_atlas_num', default=500, type=int,
                    help='maximum atlas number for each target, for random choice')
parser.add_argument('--normalization_method', default='z-score', choices=['z-score', 'min-max'], type=str,
                    help='intensity normalization method for image data')
parser.add_argument('--perform_inference', default=False, action='store_true',
                    help="whether to perform inference after model trained")
parser.add_argument('--input_scale', default=0, type=int,
                    help='the scale of the network inputs')

# training data setups
parser.add_argument('--train_target_search_path', type=str,
                    default='../../../dataset/C0T2LGE/label_center_data/training/*.png',
                    help='search pattern to find all training target images, labels and weights')
parser.add_argument('--train_atlas_search_path', type=str,
                    default='../../../dataset/C0T2LGE/label_center_data/training/*.png',
                    help='search pattern to find all training atlas images, labels and weights')
parser.add_argument('--train_image_suffix', type=str, default='image.png',
                    help='suffix pattern for the training images')
parser.add_argument('--train_label_suffix', type=str, default='label.png',
                    help='suffix pattern for the training labels')
parser.add_argument('--train_weight_suffix', type=str, default=None,
                    help='suffix pattern for the training weights')
parser.add_argument('--train_crop_patch', default=False, action='store_false',
                    help='whether patches of a certain size need to be cropped for training')
parser.add_argument('--train_patch_size', default=(144, 144), type=int, nargs=2,
                    help='size of the training patches')
parser.add_argument('--train_crop_roi', default=False, action='store_true',
                    help='whether to crop ROI containing the whole foreground on training data')
parser.add_argument('--train_channels', default=1, type=int,
                    help='number of training image channels')
parser.add_argument('--train_n_class', default=4, type=int,
                    help='number of training label classes, including the background')
parser.add_argument('--train_label_intensity', default=(0, 85, 212, 255), type=int, nargs='+',
                    help='list of intensities of the training ground truths')
parser.add_argument('--image_augment', default=False, action='store_true',
                    help='whether to apply image data augmentation on training dataset')

# data augmentation by affine transformations settings
parser.add_argument('--affine_augment', default=False, action='store_true',
                    help='whether to apply data augmentation during training')
parser.add_argument('--rot_std', default=0., type=float,
                    help='standard deviation for the rotation parameters')
parser.add_argument('--scl_std', default=0., type=float,
                    help='standard deviation for the scaling parameters')
parser.add_argument('--tra_std', default=0., type=float,
                    help='standard deviation for the translation parameters')
parser.add_argument('--she_std', default=0., type=float,
                    help='standard deviation for the shearing parameters')

# network architecture, objective function and hyper-parameter settings
parser.add_argument('--input_size', default=(144, 144), type=int, nargs=2,
                    help='the input image size for the network')
parser.add_argument('--net_method', default='ddf_label', type=str,
                    help='the method of network to infer the dense displacement fields')
parser.add_argument('--num_down_blocks', default=4, type=int,
                    help='the number of downside convolution blocks of the network')
parser.add_argument('--ddf_levels', default=None, type=int, nargs='*',
                    help='the network levels where to extract dense displacement fields')
parser.add_argument('--features_root', default=32, type=int,
                    help='number of features of the first convolution layer')
parser.add_argument('--normalizer', default='batch', type=str,
                    choices=['batch', 'group', 'layer', 'instance', 'batch_instance'],
                    help='type of network normalization method')
parser.add_argument('--gap_filling', default=False, action='store_true',
                    help='whether to use the gap-filling strategy')
parser.add_argument('--num_filling_blocks', default=(2, 1), nargs='+',
                    help='the number of filling blocks at each level')
parser.add_argument('--num_atlases', default=1, type=int,
                    help='the number of atlases fed into the network')
parser.add_argument('--num_subtypes', default=(2, 1, 1, 1), type=int, nargs='+',
                    help='A tuple indicating the number of subtypes within each tissue class, with the first element '
                         'corresponding to the background subtypes.')
parser.add_argument('--diffeomorphism', default=False, action='store_true',
                    help='whether to use diffeomorphic transformations')
parser.add_argument('--int_steps', default=4, type=int,
                    help='number of integration steps on the velocity fields')
parser.add_argument('--cost_function', default='label_consistency',
                    choices=(['mvmm', 'mvmm_mas', 'mvmm_net_gmm', 'mvmm_net_ncc', 'mvmm_net_mask', 'mvmm_net_lecc',
                              'label_consistency',  'multi_scale_label_consistency', 'dice', 'multi_scale_dice',
                              'cross_entropy', 'SSD', 'LNCC', 'KL_divergence', 'L2_norm']),
                    help='the type of cost function for network optimization')
parser.add_argument('--prior_prob', default=None, type=float, nargs='+', help='substructure prior probabilities')
parser.add_argument('--prob_method', default='use_mask', type=str, choices=['use_roi', 'use_mask', 'average', 'sum'],
                    help='whether to use certain method when computing the loss function')
parser.add_argument('--prob_sigma', default=(1, ), type=float, nargs='+',
                    help='the standard deviation of the Gaussian filter for multi-scale probability maps')
parser.add_argument('--prob_threshold', default=(0.015, 0.985), type=float, nargs='+',
                    help='probability threshold to compute mask when using label consistency loss')
parser.add_argument('--regularizer', default=(None, 'bending_energy'), nargs='+',
                    choices=([None, 'l2', 'l1', 'bending_energy']),
                    help='type of regularization, the first item for network parameters regularization, '
                         'and the second for deformation regularization')
parser.add_argument('--regularization_coefficient', default=(1e-6, 0.1), type=float, nargs='+',
                     help='regularization coefficient')
parser.add_argument('--bending_energy_increment_rate', default=1., type=float,
                    help='bending energy exponential increment rate')

# network training/validation and model restore/saving configurations
parser.add_argument('--optimizer_name', default='adam', type=str,
                    choices=['momentum', 'adam', 'sgd', 'rmsprop', 'adam-clr', 'radam', 'adabound'],
                    help='type of the optimizer to use (momentum, adam, sgd or rmsprop)')
parser.add_argument('--num_workers', default=4, type=int,
                    help='how many sub-processes to use for data loading')
parser.add_argument('--learning_rate', default=1e-5, type=float,
                    help='network learning rate')
parser.add_argument('--batch_size', default=1, type=int,
                    help='batch size for each training iteration')
parser.add_argument('--validation_batch_size', default=None, type=int,
                    help='the number of validation cases')
parser.add_argument('--restore', default=False, action='store_true',
                    help='whether previous model checkpoint need restoring')
parser.add_argument('--restore_model_path', default=None,
                    help='the path where to restore the previous model parameters')
parser.add_argument('--latest_filename', default=None, type=str,
                    help='optional name of the checkpoint file')
parser.add_argument('--model_trained_path', default='model_trained', type=str,
                    help='path where to store checkpoints')
parser.add_argument('--self_iters', default=1, type=int,
                    help='number of self iteration for each case')
parser.add_argument('--pretrain_epochs', default=0, type=int, help='number of pretraining epochs')
parser.add_argument('--pretrain_steps', default=None, type=int, help='number of pretraining steps')
parser.add_argument('--epochs', default=10, type=int, help='number of epochs')
parser.add_argument('--dropout', default=0.2, type=float, help='dropout probability')
parser.add_argument('--dropout_type', default='regular', type=str, choices=['regular', 'spatial'],
                    help='dropout type')
parser.add_argument('--clip_gradient', default=False, action='store_true',
                    help='whether to apply gradient clipping with L2 norm threshold 1.0')
parser.add_argument('--display_step', default=500, type=int,
                    help='number of steps till outputting stats')
parser.add_argument('--validation_step', default=None, type=int, help='number of steps till each validation')
parser.add_argument('--prediction_path', default='validation_prediction', type=str,
                    help='path where to save predictions on each epoch')

# validation data setups
parser.add_argument('--test_target_search_path', type=str,
                    default='../../../dataset/C0T2LGE/label_center_data/validation/*.png',
                    help='search pattern to find all test/validation images, labels and probabilities.')
parser.add_argument('--test_atlas_search_path', type=str,
                    default='../../../dataset/C0T2LGE/label_center_data/validation/*.png',
                    help='search pattern to find all test/validation fixed atlases')
parser.add_argument('--test_image_suffix', type=str, default='image.png',
                    help='suffix pattern for the test/validation images')
parser.add_argument('--test_label_suffix', type=str, default='label.png',
                    help='suffix pattern for the test/validation labels')
parser.add_argument('--test_weight_suffix', type=str, default=None,
                    help='suffix pattern for the test/validation weights')
parser.add_argument('--test_channels', default=1, type=int,
                    help='number of test/validation image channels')
parser.add_argument('--test_n_class', default=4, type=int,
                    help='number of test/validation label classes, including the background')
parser.add_argument('--test_label_intensity', default=(0, 85, 212, 255), type=int, nargs='+',
                    help='tuple of intensities of the test/validation ground truths')

parser.add_argument('--config_name', default=t + '_config.txt', type=str,
                    help='the filename to write down configurations')

args = parser.parse_args()

if __name__ == '__main__':
    # set working directory
    print("Current working directory: %s" % os.getcwd())
    os.chdir('../')
    print("Working directory changed to: %s" % os.path.abspath(os.getcwd()))

    # set cuda device
    device = '/cpu:0' if args.cuda_device == -1 else '/gpu:%s' % args.cuda_device
    # device = '/cpu:0' if args.cuda_device == -1 else '/gpu:0'
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    # set the saving directory
    if args.restore:
        # save into input restore_model_path, must be of the form: 2020-10-10_10-10-10*/trial_?/model_trained
        save_path = './%s/restore' % os.path.split(args.restore_model_path)[0]
    else:
        save_path = './%s_%s_target-%s_atlas-%s_%s' % (args.time, args.model_type,
                                                       ''.join(args.target_modalities), ''.join(args.atlas_modalities),
                                                       args.spacing)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # write configurations
    with open(os.path.join(save_path, args.config_name), 'w+') as f:
        f.write(str(args))

    # set logger
    logger = utils_2d.config_logging(os.path.join(save_path, 'log.log'))

    for i in range(args.training_times):
        # prepare data
        train_data_provider = image_utils.ImageDataProvider(target_search_path=args.train_target_search_path,
                                                            atlas_search_path=args.train_atlas_search_path,
                                                            target_modalities=args.target_modalities,
                                                            atlas_modalities=args.atlas_modalities,
                                                            a_min=args.a_min,
                                                            a_max=args.a_max,
                                                            image_suffix=args.train_image_suffix,
                                                            label_suffix=args.train_label_suffix,
                                                            weight_suffix=args.train_weight_suffix,
                                                            n_atlas=args.num_atlases,
                                                            d_baseline=args.d_baseline,
                                                            max_atlas_num=args.max_atlas_num,
                                                            crop_patch=args.train_crop_patch,
                                                            patch_size=args.train_patch_size,
                                                            crop_roi=args.train_crop_roi,
                                                            channels=args.train_channels,
                                                            n_class=args.train_n_class,
                                                            label_intensity=args.train_label_intensity,
                                                            n_subtypes=args.num_subtypes,
                                                            scale=args.input_scale,
                                                            image_augmentation=args.image_augment,
                                                            normalization_method=args.normalization_method,
                                                            logger=logger)

        test_data_provider = image_utils.ImageDataProvider(target_search_path=args.test_target_search_path,
                                                           atlas_search_path=args.test_atlas_search_path,
                                                           target_modalities=args.target_modalities,
                                                           atlas_modalities=args.atlas_modalities,
                                                           a_min=args.a_min,
                                                           a_max=args.a_max,
                                                           image_suffix=args.test_image_suffix,
                                                           label_suffix=args.test_label_suffix,
                                                           weight_suffix=args.test_weight_suffix,
                                                           n_atlas=args.num_atlases,
                                                           d_baseline=args.d_baseline,
                                                           max_atlas_num=args.max_atlas_num,
                                                           crop_patch=False,  # crop patches centered at the image center
                                                           patch_size=args.train_patch_size,
                                                           crop_roi=False,
                                                           channels=args.test_channels,
                                                           n_class=args.test_n_class,
                                                           label_intensity=args.test_label_intensity,
                                                           n_subtypes=args.num_subtypes,
                                                           scale=args.input_scale,
                                                           normalization_method=args.normalization_method,
                                                           logger=logger)

        with tf.Graph().as_default():
            # establish model
            net = model.UnifiedMultiAtlasSegNet(# basic model setups
                                                input_size=args.input_size,
                                                channels=args.train_channels,
                                                n_class=args.train_n_class,
                                                n_atlas=args.num_atlases,
                                                n_subtypes=args.num_subtypes,
                                                # network configuration
                                                method=args.net_method,
                                                num_down_blocks=args.num_down_blocks,
                                                dropout_type=args.dropout_type,
                                                ddf_levels=args.ddf_levels,
                                                features_root=args.features_root,
                                                normalizer=args.normalizer,
                                                gap_filling=args.gap_filling,
                                                num_filling_blocks=args.num_filling_blocks,
                                                diffeomorphism=args.diffeomorphism,
                                                int_steps=args.int_steps,
                                                # cost function settings
                                                cost_kwargs={'cost_name': args.cost_function,
                                                             'prior_prob': args.prior_prob,
                                                             'prob_method': args.prob_method,
                                                             'prob_sigma': args.prob_sigma,
                                                             'prob_threshold': args.prob_threshold,
                                                             'regularizer': args.regularizer,
                                                             'regularization_coefficient': args.regularization_coefficient,
                                                             'bending_energy_increment_rate': args.bending_energy_increment_rate},
                                                # data augmentation settings
                                                aug_kwargs={'affine_augment': args.affine_augment,
                                                            'rot_std': args.rot_std,
                                                            'scl_std': args.scl_std,
                                                            'tra_std': args.tra_std,
                                                            'she_std': args.she_std},
                                                logger=logger)

            # trainer initialization
            trainer = model.Trainer(net, batch_size=args.batch_size, optimizer_name=args.optimizer_name,
                                    learning_rate=args.learning_rate, num_workers=args.num_workers)

            # train network
            # Todo: check the restore model part
            trainer.train(train_data_provider, test_data_provider, args.validation_batch_size,
                          save_model_path=os.path.join(save_path, 'trial_%s' % i, args.model_trained_path),
                          pretrain_epochs=args.pretrain_epochs,
                          pretrain_steps=args.pretrain_steps,
                          epochs=args.epochs,
                          dropout=args.dropout,
                          clip_gradient=args.clip_gradient,
                          display_step=args.display_step,
                          validation_step=args.validation_step,
                          self_iters=args.self_iters,
                          prediction_path=os.path.join(save_path, 'trial_%s' % i, args.prediction_path),
                          restore=args.restore,
                          restore_model_path=args.restore_model_path,
                          latest_filename=args.latest_filename)

        # print network and trainer configuration
        logger.info(str(trainer))

        # perform inference
        if args.perform_inference:
            print("##############################################################################################")
            logging.info("Start registration on test target data!")
            os.system('python -u ./save_prediction_2d.py '
                      '--method %s '
                      '--model_path %s '
                      '--cuda_device %s '
                      '--ddf_levels %s '
                      '--features_root %s '
                      '--num_down_blocks %s' % (args.method, save_path, args.cuda_device,
                                                ' '.join(args.ddf_levels), args.features_root,
                                                args.num_down_blocks)
                      )

