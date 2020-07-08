import os
import glob
from PIL import Image
import numpy as np
from scipy import stats
import tensorflow as tf
from core import losses_2d, utils_2d


config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth = True


def _load_image(name, dtype=np.float32):
    img = Image.open(name)
    return np.asarray(img, dtype)


def _process_image(data):
    data_norm = stats.zscore(data, axis=None, ddof=1)
    return np.expand_dims(np.expand_dims(data_norm, -1), 0)


def _process_label(data, intensities=(0, 255)):
    n_class = len(intensities)
    label = np.zeros((np.hstack((data.shape, n_class))), dtype=np.float32)

    for k in range(1, n_class):
        label[..., k] = (data == intensities[k])

    label[..., 0] = np.logical_not(np.sum(label[..., 1:], axis=-1))
    return np.expand_dims(label, 0)


def _grayscale(image):
    return ((image - image.min()) / image.ptp() * 255).astype(np.uint8)


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    data_path = '../../../../../../dataset/C0T2LGE/label_center_data/training/image_slices'

    import time

    os.chdir(data_path)
    print(os.getcwd())

    image_names = glob.glob('*image.png')
    image_suffix = 'image.png'
    label_suffix = 'label.png'

    save_path = './ncc_images'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    label_intensities = (0, 85, 212, 255)

    image_tensor = tf.placeholder(tf.float32, [1, 144, 144, 1])
    label_tensor = tf.placeholder(tf.float32, [1, 144, 144, len(label_intensities)])

    # compute gradient image of intensity and label data
    label_grad = utils_2d.compute_gradnorm_from_volume(label_tensor)
    image_grad = tf.reduce_sum(utils_2d.compute_gradnorm_from_volume(image_tensor), axis=-1, keepdims=True)

    # compute local normalized cross-correlation maps from gradient images
    NCC = losses_2d.CrossCorrelation(win=7)
    ncc_tensor = tf.exp(tf.concat([NCC.ncc(image_grad, label_grad[..., i, None])
                                   for i in range(len(label_intensities))], axis=-1))

    with tf.Session(config=config) as sess:
        for name in image_names:
            print(name)
            time_start = time.time()
            image = _load_image(name)
            label = _load_image(name.replace(image_suffix, label_suffix))

            # pre-processing
            image_data = _process_image(image)
            label_data = _process_label(label, label_intensities)

            ncc = sess.run(ncc_tensor, feed_dict={image_tensor: image_data,
                                                  label_tensor: label_data})

            # for i in range(len(label_intensities)):
            #     ncc_img = Image.fromarray(_grayscale(ncc[0, ..., i]))
            #     ncc_img.show()

            print("NCC percentage: %.4f" % (np.sum(ncc > 1) / np.prod(ncc.shape)))

            np.save(os.path.join(save_path, name.replace(image_suffix, 'ncc.npy')), ncc.squeeze(0))

            time_end = time.time()
            print("Elapsing time: %s" % (time_end - time_start))
