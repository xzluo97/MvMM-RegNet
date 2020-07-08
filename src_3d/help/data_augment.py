# -*- coding: utf-8 -*-
"""
Data augmentation on training dataset.

@author: Xinzhe Luo

@version: 0.1
"""

import tensorflow as tf
import nibabel as nib
import numpy as np
from skimage import img_as_float, img_as_uint
from skimage.exposure import equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid
from skimage.util import random_noise
from skimage.filters import gaussian
import random
import math
import logging
import os
import glob
import re
import sys
sys.path.append('..')
# sys.setrecursionlimit(10**7)
from core import layers

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    """
    alist.sort(key=natural_keys) sorts in human order
    """
    return [atoi(c) for c in re.split('(\d+)', text)]


def strsort(alist):
    alist.sort(key=natural_keys)
    return alist

def randRange(a, b):
    """
    a utility function to generate random float values in desired range
    """
    return np.random.rand() * (b - a) + a


def randomIntensity(im):
    """
    rescales the intensity of the image to random interval of image intensity distribution
    """
    im = im.astype(np.int16)
    return rescale_intensity(im,
                             in_range=tuple(np.percentile(im, (randRange(0,5), randRange(95,100)))),
                             out_range=tuple(np.percentile(im, (randRange(0,5), randRange(95,100)))))


def randomGamma(im):
    '''
    Gamma filter for contrast adjustment with random gamma value.
    '''
    im = im.astype(np.int16)
    return adjust_gamma(im, gamma=randRange(0.5, 1.5))


def randomGaussian(im):
    '''
    Gaussian filter for blurring the image with random variance.
    '''
    im = im.astype(np.float32)
    return gaussian(im, sigma=randRange(0, 2), multichannel=False)


def randomNoise(im):
    '''
    random gaussian noise with random variance.
    '''
    im = im.astype(np.int16)
    var = randRange(1e-6, 1e-5)
    return random_noise(im, var=var) * im.max() + im.min()


def normalize(im):
    return (im - im.min()) / im.max()


def randomFilter(im, dtype=np.float32):
    """
    randomly selects an exposure filter from histogram equalizers, contrast adjustments, and intensity rescaling and applies it on the input image.
    filters include: equalize_hist, rescale_intensity, adjust_gamma, adjust_log, adjust_sigmoid, gaussian
    """
    Filters = [  randomGaussian, randomIntensity, randomNoise]
    filter = random.choice(Filters)
    return np.asarray(filter(im), dtype=dtype)


def equalizeHist(im):
    im = im.astype(np.int16)
    return equalize_hist(im) * np.max(im) + np.min(im)


def randomLog(im):
    im = im.astype(np.int16)
    return adjust_log(im, gain=randRange(0.8, 1.2))


def randomSigmoid(im):
    im = im.astype(np.int16)
    return adjust_sigmoid(im, cutoff=randRange(0.4, 0.6), gain=randRange(8, 12))


def randomFFD(img_name, ffd_type=1, random_type=0, control_points=(20, 20, 20), num_samples=5, **kwargs):
    """
    Generate random FFD-augmented samples using zxhInitPreTransform.exe

    :param img_name: image data file name
    :param ffd_type: '1' for world coordinate or '2' for image coordinate
    :param random_type: '0' for normal distribution or '1' for uniform distribution
    :param control_points: physical spacing if ffd_type=1; numbers if ffd_type=2
    :param num_samples: number of random augmented samples to draw
    :param kwargs: optional arguments: image_suffix, label_suffix, save_path, u, s
    """
    img_name = os.path.basename(img_name)
    image_suffix = kwargs.pop('image_suffix', 'image.nii.gz')
    label_suffix = kwargs.pop('label_suffix', 'label.nii.gz')
    lab_name = img_name.replace(image_suffix, label_suffix)
    save_path = kwargs.pop('save_path', './FFD_augmented')
    resave2int = kwargs.pop('resave2int', False)

    if ffd_type == 1:
        fn = 'ffd'
    elif ffd_type == 2:
        fn = 'ffd2'
    else:
        raise ValueError("Unknown FFD type: %s!" % ffd_type)

    if random_type == 0:
        u = kwargs.pop('mu', 0)
        s = kwargs.pop('sigma', 1)
    elif random_type == 1:
        u = kwargs.pop('a', -1)
        s = kwargs.pop('b', 1)
    else:
        raise ValueError("Unknown random type: %s!" % random_type)

    for i in range(num_samples):
        ffd_marker = os.path.join(save_path, img_name.replace(image_suffix, 'ffd%s' % i))
        if resave2int:
            os.system('zxhimageop -float {img_name:s} -toi'.format(img_name=img_name))
        # generate FFD
        os.system('zxhInitPreTransform {ffd_name} -{fn:s} {img_name:s} {spx:d} {spy:d} {spz:d} '
                  '-genrandom {random_type:d} {u:.1f} {s:.1f}'.format(ffd_name=ffd_marker, fn=fn, img_name=img_name,
                                                                      spx=control_points[0],
                                                                      spy=control_points[1],
                                                                      spz=control_points[2],
                                                                      random_type=random_type,
                                                                      u=u, s=s)
                  )
        # transform image
        output_marker = os.path.join(save_path, img_name.replace(image_suffix, 'ffd%s_' % i))
        os.system('zxhtransform {target} {source} '
                  '-o {output} -n 1 -t {ffd_name}'.format(target=img_name, source=img_name,
                                                          output=output_marker + image_suffix,
                                                          ffd_name=ffd_marker + '.FFD')
                  )
        # transform label
        os.system('zxhtransform {target} {source} '
                  '-o {output} -n 1 -t {ffd_name} -nearest'.format(target=lab_name, source=lab_name,
                                                                   output=output_marker + label_suffix,
                                                                   ffd_name=ffd_marker + '.FFD')
                  )


class DataAugmentation(object):
    def __init__(self, data_search_path, image_suffix, label_suffix, **kwargs):
        self.image_suffix = image_suffix
        self.label_suffix = label_suffix
        self.image_names = strsort(self._find_image_names(data_search_path))
        self.kwargs = kwargs
        self.affine_augment = kwargs.pop('affine_augment', False)
        
    def _init(self):
        data_size = self.kwargs.pop('data_size', (112, 96, 112))
        # placeholder for image and label
        self.data = {'image': tf.placeholder(tf.float32, [1, data_size[0], data_size[1], data_size[2], 1]),
                     'label': tf.placeholder(tf.float32, [1, data_size[0], data_size[1], data_size[2], 1])}

        self.augmented_data = self._get_augmented_data()

    def _find_image_names(self, data_search_path):
        names = glob.glob(data_search_path)
        return [name for name in names if self.image_suffix in name]

    def _get_augmented_data(self):
        augmented_data = dict(zip(['image', 'label'],
                                  layers.random_affine_augment([self.data['image'], self.data['label']],
                                                                  interp_methods=['linear', 'nearest'],
                                                                  **self.kwargs)))
        return augmented_data

    def load_data_numpy(self, name, dtype=np.float32, expand=True):
        img = nib.load(name)
        image = np.asarray(img.get_fdata(), dtype)
        if expand:
            image = np.expand_dims(image, 0)
            image = np.expand_dims(image, -1)

        return image, img.affine, img.header

    @staticmethod
    def save_into_nii(array, save_path, save_name, **kwargs):
        image = np.squeeze(array, (0, -1))
        affine = kwargs.pop('affine', np.eye(4))
        header = kwargs.pop('header', None)
        img = nib.Nifti1Image(image.astype(np.int16), affine=affine, header=header)
        nib.save(img, os.path.join(save_path, save_name))

    def augment(self, num_samples=1, save_path='./augmented_data'):
        if not os.path.exists(save_path):
            logging.info("Allocating '%s'" % os.path.abspath(save_path))
            os.makedirs(save_path)
        
        with tf.Session(config=config) as sess:
            # initialization
            if self.affine_augment:
                self._init()
            # augmentation
            for image_name in self.image_names:
                label_name = image_name.replace(self.image_suffix, self.label_suffix)
                image, affine, header = self.load_data_numpy(image_name)
                label = self.load_data_numpy(label_name)[0]

                logging.info('Augmenting data: %s' % os.path.basename(image_name))
                for i in range(num_samples):
                    # logging.info('')
                    if self.affine_augment:
                        image_affine, label_affine = sess.run((self.augmented_data['image'], self.augmented_data['label']),
                                                              feed_dict={self.data['image']: image,
                                                                         self.data['label']: label})
                        # random affine
                        self.save_into_nii(image_affine, save_path,
                                           save_name='aug%s_affine_' % i + os.path.basename(image_name),
                                           affine=affine, header=header)
                        self.save_into_nii(label_affine, save_path,
                                           save_name='aug%s_affine_' % i + os.path.basename(label_name),
                                           affine=affine, header=header)

                    # random-type augmentation
                    self.save_into_nii(randomFilter(image), save_path,
                                       save_name='aug%s_random_' % i + os.path.basename(image_name),
                                       affine=affine, header=header)

                    # # equalize histogram
                    # self.save_into_nii(equalizeHist(image), save_path,
                    #                    save_name='aug%s_equal_' % i + os.path.basename(image_name),
                    #                    affine=affine, header=header)
                    # # random gamma
                    # self.save_into_nii(randomGamma(image), save_path,
                    #                    save_name='aug%s_gamma_' % i + os.path.basename(image_name),
                    #                    affine=affine, header=header)
                    # # random Gaussian
                    # self.save_into_nii(randomGaussian(image), save_path,
                    #                    save_name='aug%s_gaussian_' % i + os.path.basename(image_name),
                    #                    affine=affine, header=header)
                    # # random intensity
                    # self.save_into_nii(randomIntensity(image), save_path,
                    #                    save_name='aug%s_int_' % i + os.path.basename(image_name),
                    #                    affine=affine, header=header)
                    # # random noise
                    # self.save_into_nii(randomNoise(image), save_path,
                    #                    save_name='aug%s_noise_' % i + os.path.basename(image_name),
                    #                    affine=affine, header=header)
                    # # logarithmic correction
                    # self.save_into_nii(randomLog(image), save_path,
                    #                    save_name='aug%s_log_' % i + os.path.basename(image_name),
                    #                    affine=affine, header=header)
                    # # sigmoid correction
                    # self.save_into_nii(randomSigmoid(image), save_path,
                    #                    save_name='aug%s_sigmoid_' % i + os.path.basename(image_name),
                    #                    affine=affine, header=header)

            logging.info("Augmentation Finished!")


class FFDAugmentation(DataAugmentation):
    """
    Data augmentation by free-form deformation using zxhInitPreTransform.exe.

    """
    def __init__(self, data_search_path, image_suffix, label_suffix, **kwargs):
        super(FFDAugmentation, self).__init__(data_search_path, image_suffix, label_suffix, **kwargs)
        print(self.image_names)
        
    def augment(self, num_samples=5, save_path='./FFD_augmented', **kwargs):
        if not os.path.exists(save_path):
            logging.info("Allocating '%s'" % os.path.abspath(save_path))
            os.makedirs(save_path)
            
        for image_name in self.image_names:
            logging.info('Augmenting data: %s' % os.path.basename(image_name))
            randomFFD(image_name, num_samples=num_samples, save_path=save_path,
                      image_suffix=self.image_suffix, label_suffix=self.label_suffix, **kwargs)
        
        logging.info("Augmentation Finished!")


if __name__ == '__main__':
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    os.environ['CUDA_VISIBLE_DEVICE'] = '-1'
     
    data_path = ''

    logging.info("Current working directory: %s" % os.path.abspath(os.getcwd()))
    os.chdir(data_path)
    logging.info("Working directory changed to %s" % os.path.abspath(os.getcwd()))

    # DA = DataAugmentation(data_search_path='./FFD_augmented_20/*.nii.gz',
    #                       image_suffix='image.nii.gz', label_suffix='label.nii.gz',
    #                       rot_std=math.pi/36, tra_std=0., scl_std=0.05, she_std=0.05)
    # DA.augment(num_samples=1, save_path='./FFD_augmented_20/augmented_data')
    
    FA = FFDAugmentation(data_search_path='./*.nii.gz', image_suffix='image.nii.gz', label_suffix='label.nii.gz')
    FA.augment(num_samples=1, save_path='./FFD_augmented_200', ffd_type=1, control_points=(10, 10, 10),
               random_type=0, mu=0, sigma=5, resave2int=True)
