import os
import glob
import nibabel as nib
import numpy as np
from core import utils


def load_image(name, dtype=np.float32):
    img = nib.load(name)
    return np.asarray(img.get_fdata(), dtype), img.affine, img.header


def process_label(data, intensities=(0, 205)):
    n_class = len(intensities)
    label = np.zeros((np.hstack((data.shape, n_class))), dtype=np.float32)

    for k in range(1, n_class):
        label[..., k] = (data == intensities[k])

    label[..., 0] = np.logical_not(np.sum(label[..., 1:], axis=-1))
    return np.expand_dims(label, 0)


if __name__ == '__main__':
    data_path = '../../../../../../dataset/training_mr_40_ffd_aug_commonspace2/*label.nii.gz'

    label_names = glob.glob(data_path)
    label_suffix = 'label.nii.gz'

    os.chdir(os.path.dirname(data_path))
    print(os.getcwd())

    save_path = './mask_images'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    label_intensities = (0, 205, 420, 500, 550, 600, 820, 850)

    for name in label_names:
        label_name = os.path.basename(name)
        label, affine, header = load_image(label_name)

        label_data = process_label(label, intensities=label_intensities)

        prob_data = utils.get_prob_from_label(label_data, sigma=1., mode='np')

        mask_data = utils.compute_mask_from_prob(prob_data, mode='np')

        mask_nii = nib.Nifti1Image(mask_data.squeeze(0), affine=affine, header=header)
        nib.save(mask_nii, os.path.join(save_path, label_name.replace(label_suffix, 'mask.nii.gz')))


