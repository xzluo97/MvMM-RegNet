import os
import glob
import nibabel as nib
import numpy as np
from scipy import stats
from sklearn import mixture
from core import utils


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


class GMM(object):
    def __init__(self, n_class, n_subtypes, **kwargs):
        self.n_class = n_class
        self.n_subtypes = n_subtypes
        self.kwargs = kwargs
        # self.eps = kwargs.pop('eps', 1e-5)

    def get_gmm_coefficients(self, image, label):
        """
        Get the image mixture coefficients of each subtype within the tissue class using the label image.

        :param image: The image array of shape [1, *vol_shape, channels].
        :param label: The label array of shape [1, *vol_shape, n_class].
        :return: tau - a list of arrays of shape [n_subtypes[i]];
                 mu - a list of arrays of shape [n_subtypes[i]];
                 sigma - a list of arrays of shape [n_subtypes[i]].
        """
        tau = []
        mu = []
        sigma = []
        for i in range(self.n_class):
            image_take = np.take(np.sum(image, axis=-1).flatten(), indices=np.where(label[..., i].flatten() == 1))
            clf = mixture.GaussianMixture(n_components=self.n_subtypes[i])
            clf.fit(image_take.reshape(-1, 1))
            tau.append(clf.weights_)
            mu.append(clf.means_.squeeze(1))
            sigma.append(np.sqrt(clf.covariances_.squeeze((1, 2))))
        return tau, mu, sigma

    def get_gmm_cond_probs(self, image, tau, mu, sigma):
        """
        Compute the conditional pdf of the target image intensities given the tissue class.

        :param image: The target images, of shape [n_batch, *vol_shape, channels].
        :param tau: The mixing coefficients of Gaussian's, a list with each entry of shape [n_batch, n_subtypes[i]].
        :param mu: The means of Gaussian's, a list with each entry of shape [n_batch, n_subtypes[i]].
        :param sigma: The standard deviations of Gaussian's, a list with each entry of shape [n_batch, n_subtypes[i]].
        :return: A tensor of shape [n_batch, *vol_shape, n_class], representing the conditional probabilities given each
            class.
        """
        n_batch = image.shape[0]
        class_cond_probs = []
        for i in range(self.n_class):
            tau_expand = np.reshape(tau[i], [n_batch, 1, 1, 1, self.n_subtypes[i]])
            mu_expand = np.reshape(mu[i], [n_batch, 1, 1, 1, self.n_subtypes[i]])
            sigma_expand = np.reshape(sigma[i], [n_batch, 1, 1, 1, self.n_subtypes[i]])

            pdf = utils.gaussian_pdf_numpy(image, loc=mu_expand, scale=sigma_expand)
            # dist = tfd.Normal(loc=mu_expand, scale=sigma_expand)
            # pdf = dist.prob(tf.reduce_sum(self.target_images, axis=-1, keepdims=True), name='subtype_pdf')  # [
            # n_batch, *vol_shape, n_subtypes[i]]

            class_cond_probs.append(np.sum(tau_expand * pdf, axis=-1))  # [n_batch, *vol_shape]

        return np.stack(class_cond_probs, axis=-1)


if __name__ == '__main__':
    data_path = '../../../../../../dataset/validation_mr_5_commonspace2/*_image.nii.gz'

    image_names = glob.glob(data_path)
    image_suffix = 'image.nii.gz'
    label_suffix = 'label.nii.gz'
    label_intensities = (0, 205, 420, 500, 550, 600, 820, 850)

    import time

    os.chdir(os.path.dirname(data_path))
    print(os.getcwd())

    save_path = './gmm_grad_images'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    gmm_model = GMM(n_class=8, n_subtypes=(2, 2, 2, 2, 2, 2, 2, 2))

    for name in image_names:
        time_start = time.time()
        image_name = os.path.basename(name)
        print("image name: %s" % image_name)
        image, affine, header = load_image(image_name)
        label = load_image(image_name.replace(image_suffix, label_suffix))[0]

        # pre-processing for image and label
        image_data = process_image(image)
        image_grad = utils.compute_gradnorm_from_volume(np.sum(image_data, axis=-1, keepdims=True), mode='np')
        label_data = process_label(label, label_intensities)

        gmm_prob = gmm_model.get_gmm_cond_probs(image_grad, *gmm_model.get_gmm_coefficients(image_grad, label_data))

        gmm_nii = nib.Nifti1Image(gmm_prob.squeeze(0), affine=affine, header=header)
        nib.save(gmm_nii, os.path.join(save_path, image_name.replace(image_suffix, 'gmm_grad.nii.gz')))

        time_end = time.time()
        print("Elapsing time: %s" % (time_end - time_start))

