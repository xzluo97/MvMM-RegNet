import os
import glob
import nibabel as nib
import numpy as np
from scipy import stats
from core import utils, losses
import tensorflow as tf


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


def load_image(name, dtype=np.float32):
    img = nib.load(name)
    return np.asarray(img.get_fdata(), dtype), img.affine, img.header


def process_image(data):
    data_norm = stats.zscore(data, axis=None, ddof=1)
    return np.expand_dims(np.expand_dims(data_norm, -1), 0)


def process_label(data, intensities=(0, 205)):
    n_class = len(intensities)
    label = np.zeros((np.hstack((data.shape, n_class))), dtype=np.float32)

    for k in range(1, n_class):
        label[..., k] = (data == intensities[k])

    label[..., 0] = np.logical_not(np.sum(label[..., 1:], axis=-1))
    return np.expand_dims(label, 0)


# if __name__ == '__main__':
#
#     for name in label_names:
#         os.system(r'zxhvolumelabelop %s %s -genprob 3 3 3' % (name, name.replace('_label', '_distance_prob')))


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_path = '../../../../../../dataset/training_ct_20_commonspace2'
    os.chdir(data_path)
    print(os.getcwd())

    label_names = glob.glob('*label.nii.gz')
    image_suffix = 'image.nii.gz'
    label_suffix = 'label.nii.gz'

    import time

    save_path = './ncc_images'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    label_intensities = (0, 205, 420, 500, 550, 600, 820, 850)

    image_tensor = tf.placeholder(tf.float32, [1, 112, 96, 112, 1])
    label_tensor = tf.placeholder(tf.float32, [1, 112, 96, 112, len(label_intensities)])

    # compute gradient image of intensity and label data
    label_grad = utils.compute_gradnorm_from_volume(label_tensor)
    image_grad = tf.reduce_sum(utils.compute_gradnorm_from_volume(image_tensor), axis=-1, keepdims=True)

    # compute local normalized cross-correlation maps from gradient images
    NCC = losses.CrossCorrelation(win=5, kernel='ones')
    ncc_tensor = tf.exp(tf.concat([NCC.ncc(image_grad, label_grad[..., i, None])
                                   for i in range(len(label_intensities))], axis=-1))

    with tf.Session(config=config) as sess:
        for name in label_names:
            print(name)
            time_start = time.time()
            label_name = os.path.basename(name)
            label, affine, header = load_image(label_name)
            image = load_image(label_name.replace(label_suffix, image_suffix))[0]

            # pre-processing for image and label
            image_data = process_image(image)
            label_data = process_label(label, intensities=label_intensities)

            # produce probability maps
            # prob_data = utils.get_prob_from_label(tf.constant(label_data), sigma=1.)

            # produce masks from probability maps
            # mask_data = utils.compute_mask_from_prob(prob_data).eval()
            # print("Mask percentage: %.4f" % (np.sum(mask_data) / np.prod(mask_data.shape)))


            # evaluate tensors
            # prob_grad = prob_grad_tensor.eval()
            # image_grad = image_grad_tensor.eval()
            ncc = sess.run(ncc_tensor, feed_dict={image_tensor: image_data, label_tensor: label_data})
            print("NCC percentage: %.4f" % (np.sum(ncc > 1) / np.prod(ncc.shape)))

            # save into nifty files
            # prob = nib.Nifti1Image((prob_data.eval().squeeze(0)*1000).astype(np.uint16),
            #                        affine=affine, header=header)
            # nib.save(prob, os.path.join(save_path, label_name.replace(label_suffix, 'prob.nii.gz')))

            # mask = nib.Nifti1Image(mask_data.squeeze(0).astype(np.uint16), affine=affine, header=header)
            # nib.save(mask, os.path.join(save_path, label_name.replace(label_suffix, 'mask.nii.gz')))

            ncc_nii = nib.Nifti1Image((ncc.squeeze(0)), affine=affine, header=header)
            nib.save(ncc_nii, os.path.join(save_path, label_name.replace(label_suffix, 'ncc_exp.nii.gz')))

            # img_grad_nii = nib.Nifti1Image((image_grad.squeeze((0, -1))*1000).astype(np.uint16), affine=affine, header=header)
            # nib.save(img_grad_nii, os.path.join(save_path, 'img_grad_' + label_name))

            # prob_grad_nii = nib.Nifti1Image((prob_grad.squeeze((0, -1))*1000).astype(np.uint16), affine=affine, header=header)
            # nib.save(prob_grad_nii, os.path.join(save_path, 'prob_grad_' + label_name))

            time_end = time.time()
            print("Elapsing time: %s" % (time_end - time_start))
