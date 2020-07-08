import os
import glob
import nibabel as nib
import numpy as np
from scipy import stats
from core import utils, losses


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


if __name__ == '__main__':
    data_path = '../../../../../../dataset/validation_mr_5_commonspace2/*_image.nii.gz'

    image_names = glob.glob(data_path)
    image_suffix = 'image.nii.gz'
    label_suffix = 'label.nii.gz'
    label_intensities = (0, 205, 420, 500, 550, 600, 820, 850)

    import time

    os.chdir(os.path.dirname(data_path))
    print(os.getcwd())

    save_path = './lecc_images'
    original_size = (112, 96, 112)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    mi = losses.MutualInformation(win=7, n_bins=4)

    for name in image_names:
        time_start = time.time()
        image_name = os.path.basename(name)
        print("image name: %s" % image_name)
        image, affine, header = load_image(image_name)
        label = load_image(image_name.replace(image_suffix, label_suffix))[0]

        # pre-processing for image and label
        image_data = process_image(image)
        label_data = process_label(label, label_intensities)

        # produce gradient images
        image_grad = utils.compute_gradnorm_from_volume(np.sum(image_data, axis=-1, keepdims=True), mode='np')
        label_grad = utils.compute_gradnorm_from_volume(label_data, mode='np')

        # save into nifty files
        # grad_nii = nib.Nifti1Image(image_grad.squeeze(0), affine=affine, header=header)
        # nib.save(grad_nii, os.path.join(save_path, image_name.replace(image_suffix, 'image_grad.nii.gz')))
        # grad_nii = nib.Nifti1Image(label_grad.squeeze(0), affine=affine, header=header)
        # nib.save(grad_nii, os.path.join(save_path, image_name.replace(image_suffix, 'label_grad.nii.gz')))

        # crop data
        image_grad_crop = utils.crop_to_shape(image_grad, (80, 80, 80))
        label_grad_crop = utils.crop_to_shape(label_grad, (80, 80, 80))

        # # compute local conditional entropy
        lecc = np.concatenate([mi.lecc(image_grad_crop, label_grad_crop[..., i, None])
                               for i in range(len(label_intensities))], axis=-1)
        # lecc = losses._lecc(image_grad_crop, label_grad_crop[..., 0, None], n_bins=4, win=7, sigma=3)

        # save into nifty files
        lecc_nii = nib.Nifti1Image(utils.pad_to_shape_image(mi._normalize(-lecc),
                                                               original_size, mode='np', method='edge').squeeze(0),
                                   affine=affine, header=header)
        nib.save(lecc_nii, os.path.join(save_path, image_name.replace(image_suffix, 'lecc.nii.gz')))

        time_end = time.time()
        print("Elapsing time: %s" % (time_end - time_start))

