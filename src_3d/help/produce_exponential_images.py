import os
import glob
import nibabel as nib
import numpy as np
from core import utils


def load_image(name, dtype=np.float32):
    img = nib.load(name)
    return np.expand_dims(np.asarray(img.get_fdata(), dtype), 0), img.affine, img.header


if __name__ == '__main__':
    data_path = '../../../../../../dataset/training_ct_40_ffd_aug_commonspace2/lecc_images/*lecc.nii.gz'

    image_names = glob.glob(data_path)
    image_suffix = 'lecc.nii.gz'

    os.chdir(os.path.dirname(data_path))
    print(os.getcwd())

    save_path = '.'
    save_suffix = 'lecc_exp.nii.gz'
    original_size = (112, 96, 112)

    for name in image_names:
        image_name = os.path.basename(name)
        image, affine, header = load_image(image_name)

        image_exp = np.exp(image)

        image_pad = utils.pad_to_shape_image(image_exp, original_size, mode='np', method='edge')

        image_nii = nib.Nifti1Image(image_pad.squeeze(0), affine=affine, header=header)
        nib.save(image_nii, os.path.join(save_path, image_name.replace(image_suffix, save_suffix)))
