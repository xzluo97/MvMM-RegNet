# -*- coding: utf-8 -*-
"""
Save atlases propagation results using registration with dense displacement fields predicted from networks.

@author: Xinzhe Luo

@version: 0.1
"""


from core import model_2d_ddf_mvmm_label_base as model
from core import image_2d_dataset as image_utils
from core import utils_2d, metrics_2d
import numpy as np
import nibabel as nib
from PIL import Image
import os
import logging
import tensorflow as tf
# import pandas as pd
import argparse
import random
from itertools import chain, combinations
from datetime import datetime
import matplotlib.pyplot as plt
from skimage import measure


t = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

parser = argparse.ArgumentParser(description='Atlas propagation on test dataset!')
parser.add_argument('--time', default=t, type=str,
                    help='The current time to save test predictions.')
parser.add_argument('--seed', default=2333, type=int, help='random seed')
parser.add_argument('--save2volume', default=False, action='store_true',
                    help='whether to assemble 2D slice predictions into volume and save the volume into nifty files')
parser.add_argument('--save2fig', default=False, action='store_true',
                    help='whether to store figures of 2D slice predictions')
parser.add_argument('--spacing', default='1.5mm', type=str, choices=['1.5mm'],
                    help='The spatial spacing of the network inputs and the dense displacement fields.')
parser.add_argument('--dropout', default=0.1, type=float, help='The dropout probability for network prediction.')
parser.add_argument('--dropout_type', default='regular', type=str, choices=['regular', 'spatial'],
                    help='dropout type')
parser.add_argument('--model_path', type=str, default=None,
                    help='The model path to restore the network parameters for network prediction.')
parser.add_argument('--latest_filename', default='best_checkpoint', type=str,
                    help='latest filename to restore the model')
parser.add_argument('--trial', default=0, type=int, help='which trial to load the model')
parser.add_argument('--cuda_device', default=0, type=int,
                    help='The cuda device for network prediction.')
parser.add_argument('--target_search_path', type=str,
                    default='../../../dataset/C0T2LGE/label_center_data/test/*.png',
                    help='search pattern to find all test target images, labels and weights')
parser.add_argument('--atlas_search_path',
                    default='../../../dataset/C0T2LGE/label_center_data/test/*.png',
                    help='The search pattern to find all training atlas images, labels and probabilities.')
parser.add_argument('--target_modalities', default=('DE', ), choices=["C0", "DE", "T2"], nargs='+',
                    help="the modalities of target images, among 'C0', 'DE' and 'T2'")
parser.add_argument('--atlas_modalities', default=('C0', ), choices=['C0', 'DE', 'T2'], nargs='+',
                    help="the modalities of atlas images, among 'C0', 'DE' and 'T2'")
parser.add_argument('--a_min', default=None, type=float, help='min value for intensity clipping')
parser.add_argument('--a_max', default=None, type=float, help='max value for intensity clipping')
parser.add_argument('--d_baseline', default=10, type=float,
                    help='slices within at least d_baseline distance will be taken into account as image pairs')
parser.add_argument('--max_atlas_num', default=1000, type=int,
                    help='maximal number of atlases for each target, for random choice')
parser.add_argument('--num_atlases', default=1, type=int, help='number of atlases for groupwise reggistration')
parser.add_argument('--fusion_atlases', default=1, type=int, help='number of atlases for fusion')
parser.add_argument('--max_pair_num', default=20, type=int, help='maximal number of groupwise atlases for each target')
parser.add_argument('--image_suffix', default='image.png', type=str,
                    help='suffix pattern for the images')
parser.add_argument('--label_suffix', default='label.png', type=str,
                    help='suffix pattern for the labels')
parser.add_argument('--weight_suffix', default=None, type=None,
                    help='suffix pattern for the weights')
parser.add_argument('--crop_patch', default=False, type=bool,
                    help='whether to crop patches of the test data')
parser.add_argument('--patch_center', default=None, nargs='+',
                    help='The customized patch center, default is None.')
parser.add_argument('--patch_size', default=(144, 144), type=int, nargs=2,
                    help='The size of the cropped patch.')
parser.add_argument('--original_size', default=(144, 144), type=int, nargs=2,
                    help='original size of the saved image')
parser.add_argument('--method', default='ddf_label',
                    choices=['ddf_label'], type=str,
                    help='the method of network to infer the dense displacement fields')
parser.add_argument('--ddf_levels', default=None, type=int, nargs='*',
                    help='the network levels where to extract dense displacement fields')
parser.add_argument('--features_root', default=32, type=int,
                    help='number of features of the first convolution layer')
parser.add_argument('--normalizer', default='instance', type=str,
                    choices=['batch', 'group', 'layer', 'instance', 'batch_instance'],
                    help='type of network normalization method')
parser.add_argument('--cost_function', default='mvmm_net_ncc',
                    choices=(['mvmm_net_mask']),
                    help='the type of cost function for network optimization')
parser.add_argument('--gap_filling', default=False, action='store_true',
                    help='whether to use the gap-filling strategy')
parser.add_argument('--num_filling_blocks', default=(2, 1), nargs='+',
                    help='the number of filling blocks at each level')
parser.add_argument('--save_ddf', default=False, action='store_true',
                    help='whether to save displacement field into nifty files')
parser.add_argument('--fig_shape', default=(100, 100), nargs=2, type=int, help='shape for saving figures')
# parser.add_argument('--save_path', default='./', type=str,
#                     help="Path where to save the test results.")
args = parser.parse_args()


def crop_center(data, shape):
    """

    :param data: of shape [nx, ny, ...]
    :param shape: [nx, ny]
    :return:
    """
    if shape is None:
        return data

    data_shape = data.shape

    offset0 = (data_shape[0] - shape[0]) // 2
    offset1 = (data_shape[1] - shape[1]) // 2
    remainder0 = (data_shape[0] - shape[0]) % 2
    remainder1 = (data_shape[1] - shape[1]) % 2

    if (data_shape[0] - shape[0]) == 0 and (data_shape[1] - shape[1]) == 0:
        return data

    elif (data_shape[0] - shape[0]) != 0 and (data_shape[1] - shape[1]) == 0:
        return data[offset0:(-offset0 - remainder0), ]

    elif (data_shape[0] - shape[0]) == 0 and (data_shape[1] - shape[1]) != 0:
        return data[:, offset1:(-offset1 - remainder1), ]

    else:
        return data[offset0:(-offset0 - remainder0), offset1:(-offset1 - remainder1), ]


def save_img_seg(ax, image, label, **kwargs):
    """
    plot image and label into figure

    :param ax:
    :param image: image of shape [nx, ny]
    :param label: label of shape [nx, ny, n_atlas, n_class]
    :param kwargs:
    :return:
    """
    shape = kwargs.pop('shape', None)
    label_intensity = kwargs.pop('label_intensity', (0, 200, 500, 600))
    intensity = np.tile(np.asarray(label_intensity), np.concatenate([label.shape[:-1], [1]]))
    label = np.sum(label * intensity, axis=-1)  # [nx, ny, n_atlas]
    label = label.transpose((1, 0, 2))
    pigment = {200: 'silver', 500: 'red', 600: 'skyblue'}
    n_atlas = label.shape[-1]
    ax.imshow(crop_center(image, shape), cmap='gray')
    label = crop_center(label, shape)
    for i in range(n_atlas):
        for k, v in pigment.items():
            contours = measure.find_contours(label[..., i] == k, 0.5, 'high')

            for c in contours:
                ax.plot(c[:, 0], c[:, 1], color=v, linewidth=2)

    ax.set_xticks([])
    ax.set_yticks([])


def _normalize_intensity(img):
    img -= np.mean(img)
    img /= np.std(img)
    return img


def compute_intensity_difference(t_name, a_names):
    t_img = np.asarray(Image.open(t_name), np.float32)
    a_imgs = [np.asarray(Image.open(name), np.float32) for name in a_names]

    t_img = np.clip(t_img, -np.inf, np.percentile(t_img, 99))
    a_imgs = [np.clip(img, -np.inf, np.percentile(img, 99)) for img in a_imgs]

    t_img = _normalize_intensity(t_img)
    a_imgs = [_normalize_intensity(img) for img in a_imgs]

    return np.mean([np.sqrt(np.sum((t_img - img) ** 2)) for img in a_imgs])


if __name__ == '__main__':

    assert args.fusion_atlases % args.num_atlases == 0, "Number of atlases for fusion must be divisible by " \
                                                        "number of atlases for groupwise registration!"

    group_reg_times = args.fusion_atlases // args.num_atlases

    # set random seed
    random.seed(args.seed)

    # set cuda device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_device)

    # set working directory
    logging.info("Current working directory: %s" % os.getcwd())
    os.chdir('../')
    logging.info("Working directory changed to: %s" % os.path.abspath(os.getcwd()))

    save_path = os.path.join(args.model_path,
                             'test_predictions_%s_group%s_fusion%s' % (args.spacing,
                                                                       args.num_atlases, args.fusion_atlases))
    metrics_path = os.path.join(args.model_path, 'metrics_test_%s_group%s_fusion%s' % (args.spacing,
                                                                                       args.num_atlases,
                                                                                       args.fusion_atlases))
    if not os.path.exists(save_path):
        logging.info("Allocating '{:}'".format(save_path))
        os.makedirs(save_path)

    if 'model_trained' in args.model_path and 'trial' in args.model_path:
        model_path = args.model_path
    else:
        model_dir = os.path.join(args.model_path, 'trial_%s' % args.trial, 'model_trained')
        ckpt = tf.train.get_checkpoint_state(model_dir, latest_filename=args.latest_filename)
        model_path = ckpt.model_checkpoint_path

    test_data_provider = image_utils.ImageDataProvider(target_search_path=args.target_search_path,
                                                       atlas_search_path=args.atlas_search_path,
                                                       target_modalities=args.target_modalities,
                                                       atlas_modalities=args.atlas_modalities,
                                                       a_min=args.a_min,
                                                       a_max=args.a_max,
                                                       image_suffix=args.image_suffix,
                                                       label_suffix=args.label_suffix,
                                                       weight_suffix=args.weight_suffix,
                                                       n_atlas=args.num_atlases,
                                                       d_baseline=args.d_baseline,
                                                       max_atlas_num=args.max_atlas_num,
                                                       crop_patch=args.crop_patch,
                                                       patch_size=args.patch_size,
                                                       crop_roi=False,
                                                       channels=1,
                                                       n_class=4,
                                                       label_intensity=(0, 85, 212, 255),
                                                       n_subtypes=(2, 1, 1, 1))

    logging.info("Number of test target-atlases pairs: %s" % len(test_data_provider))

    device = '/cpu:0' if args.cuda_device == -1 else '/gpu:%s' % args.cuda_device
    with tf.Graph().as_default():
        net = model.NetForPrediction(input_size=args.original_size,
                                     channels=1, n_class=4, n_subtypes=(2, 1, 1, 1),
                                     n_atlas=args.num_atlases,
                                     method=args.method,
                                     features_root=args.features_root,
                                     normalizer=args.normalizer,
                                     num_down_blocks=4,
                                     dropout_type=args.dropout_type,
                                     ddf_levels=args.ddf_levels,
                                     gap_filling=args.gap_filling,
                                     num_filling_blocks=args.num_filling_blocks,
                                     cost_kwargs={'cost_name': args.cost_function,
                                                  'regularizer': [None, 'bending_energy'],
                                                  'regularization_coefficient': [0., 0.]})

        # get all target names and patient IDs
        all_target_names = utils_2d.remove_duplicates([pair_names[0]
                                                       for pair_names in test_data_provider.target_atlas_image_names])
        all_patient_ids = utils_2d.remove_duplicates([os.path.basename(name).split('_')[0] for name in all_target_names])
        # split target names according to patient ID and slice position
        all_target_names = [sorted([name for name in all_target_names if os.path.basename(name).split('_')[0] == id],
                                    key=lambda x: int(os.path.basename(x).split('_')[2])) for id in sorted(all_patient_ids)]

        # list the metrics that need reporting
        test_metrics = {'Dice': [], 'Jaccard': [], 'Myocardial Dice': [], 'LV Dice': [], 'RV Dice': []}

        Dice = metrics_2d.OverlapMetrics(n_class=4, mode='np')

        with tf.Session(config=config) as sess:
            # initialize weights
            sess.run(tf.global_variables_initializer())

            # restore parameters from saved model
            net.restore(sess, model_path, var_list=net.variables_to_restore)

            # range over all target slice names
            for target_slice_names in all_target_names:
                target_predictions = []  # list to store slice predictions

                for slice_name in target_slice_names:
                    # list all atlases for t_name
                    atlas_names_init = [pair_names[1] for pair_names in test_data_provider.target_atlas_image_names
                                        if pair_names[0] == slice_name]
                    # # randomly shuffle the list
                    # random.shuffle(atlas_names_init)

                    # random select and sort atlases pairs according to intensity difference
                    # atlas_names = sorted(random.sample(atlas_names_init, max(len(atlas_names_init) // 4, group_reg_times)),
                    #                      key=lambda a_names: compute_intensity_difference(slice_name, a_names))

                    # screen out disjoint atlases pairs
                    # for a_names in atlas_names_init:
                    #     if len({*a_names}.union(set(chain(*atlas_names)))) == len(a_names) + len(list(chain(*atlas_names))):
                    #         atlas_names.append(a_names)

                    #
                    atlas_names = random.sample(atlas_names_init, max(len(atlas_names_init) // 4, group_reg_times))

                    # sample atlases pairs if the number is too large
                    # while utils_2d.nCr(len(atlas_names), args.fusion_atlases // args.num_atlases) > 1e8:
                    #     atlas_names = random.sample(atlas_names, len(atlas_names) // 2)

                    # make combinations of atlases pairs w.r.t the times of groupwise registration
                    assert len(atlas_names) >= group_reg_times, "Number of atlases pairs for " \
                                                                "target slice %s should be " \
                                                                "no less than the times of groupwise " \
                                                                "registration %s, got %s!" % (os.path.basename(slice_name),
                                                                                              group_reg_times,
                                                                                              len(atlas_names))
                    # comb_atlas_names = list(combinations(atlas_names, args.fusion_atlases // args.num_atlases))

                    comb_atlas_names = [atlas_names[i:i+group_reg_times] for i in range(0, len(atlas_names), group_reg_times)]

                    # select the atlases pair with the smallest intensity difference if save2volume
                    if args.save2volume:
                        comb_atlas_names = comb_atlas_names[:1]

                    # restrain the total number of atlases pairs for each target
                    if len(comb_atlas_names) > args.max_pair_num:
                        comb_atlas_names = comb_atlas_names[:args.max_pair_num]

                    # groupwise registration for each atlases pairs
                    for comb_a_names in comb_atlas_names:
                        # store test predictions
                        warped_atlases_weights = []
                        warped_atlases_probs = []

                        for a_names in comb_a_names:
                            logging.info("Target: %s, Atlases: %s" % (os.path.basename(slice_name),
                                                                      [os.path.basename(a_name) for a_name in a_names]))
                            # get test data index
                            idx = test_data_provider.target_atlas_image_names.index((slice_name, a_names))

                            # get test data
                            test_data = test_data_provider[idx]

                            # predict
                            warped_atlases_prob, warped_atlases_weight = net.predict_scale(sess, test_data, args.dropout)

                            warped_atlases_weights.append(warped_atlases_weight)
                            warped_atlases_probs.append(warped_atlases_prob)

                        # get fusion predictor
                        slice_predictor = utils_2d.get_segmentation(
                            utils_2d.get_joint_prob(np.concatenate(warped_atlases_probs,
                                                                   axis=-2) * np.concatenate(warped_atlases_weights,
                                                                                             axis=-2),
                                                    mode='np'), mode='np')
                        # save into figure
                        if args.save2fig:
                            fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                            save_img_seg(ax[0], test_data['target_image'].squeeze((0, -1)),
                                         np.expand_dims(test_data['target_label'].squeeze(0), -2),
                                         shape=args.fig_shape)
                            save_img_seg(ax[1], test_data['target_image'].squeeze((0, -1)),
                                         test_data['atlases_label'].squeeze(0),
                                         shape=args.fig_shape)
                            save_img_seg(ax[2], test_data['target_image'].squeeze((0, -1)),
                                         np.expand_dims(slice_predictor.squeeze(0), axis=-2),
                                         shape=args.fig_shape)

                            fig.savefig(os.path.join(save_path,
                                                     'target-%s_atlases-%s.png' % ('_'.join(os.path.basename(slice_name).split('_')[:3]),
                                                                                  '+'.join(['_'.join(os.path.basename(a_name).split('_')[:3])
                                                                                            for a_name in a_names])
                                                                                  )
                                                     ), bbox_inches='tight')

                            plt.close(fig)

                        # store metrics
                        dice = Dice.averaged_foreground_dice(test_data['target_label'], slice_predictor)
                        jaccard = Dice.averaged_foreground_jaccard(test_data['target_label'], slice_predictor)
                        myo_dice = Dice.class_specific_dice(test_data['target_label'], slice_predictor, 1)
                        LV_dice = Dice.class_specific_dice(test_data['target_label'], slice_predictor, 2)
                        RV_dice = Dice.class_specific_dice(test_data['target_label'], slice_predictor, 3)

                        logging.info("Dice= {:.4f}, Jaccard= {:.4f}, Myocardial Dice= {:.4f}, "
                                     "LV Dice= {:.4f}, RV Dice= {:.4f}".format(dice, jaccard, myo_dice, LV_dice, RV_dice))

                        test_metrics['Dice'].append(dice)
                        test_metrics['Jaccard'].append(jaccard)
                        test_metrics['Myocardial Dice'].append(myo_dice)
                        test_metrics['LV Dice'].append(LV_dice)
                        test_metrics['RV Dice'].append(RV_dice)

                    # dye label and store target slice prediction (rotate 90)
                    slice_predictor = slice_predictor.squeeze(0)
                    intensity = np.tile(np.asarray((0, 200, 500, 600)), np.concatenate((slice_predictor.shape[:-1], [1])))
                    target_predictions.append(np.rot90(np.sum(slice_predictor * intensity, axis=-1), k=-1))

                # save target volume predictions
                if args.save2volume:
                    lab = nib.Nifti1Image(np.stack(target_predictions, axis=2).astype(np.uint16), affine=np.eye(4))
                    nib.save(lab, os.path.join(save_path, 'pred_' +
                                               '_'.join(os.path.basename(target_slice_names[0]).split('_')[:2]) +
                                               '_label.nii.gz'))

        # report average metrics statistics
        logging.info("############ Average Test Statistics ############")
        logging.info("Average Dice= {:.4f}+-{:.4f}, Jaccard= {:.4f}+-{:.4f}, Myocardial Dice= {:.4f}+-{:.4f}, "
                     "LV Dice= {:.4f}+-{:.4f}, RV Dice= {:.4f}+-{:.4f}".format(np.mean(test_metrics['Dice']),
                                                                               np.std(test_metrics['Dice']),
                                                                               np.mean(test_metrics['Jaccard']),
                                                                               np.std(test_metrics['Jaccard']),
                                                                               np.mean(test_metrics['Myocardial Dice']),
                                                                               np.std(test_metrics['Myocardial Dice']),
                                                                               np.mean(test_metrics['LV Dice']),
                                                                               np.std(test_metrics['LV Dice']),
                                                                               np.mean(test_metrics['RV Dice']),
                                                                               np.std(test_metrics['RV Dice'])
                                                                               )
                     )
