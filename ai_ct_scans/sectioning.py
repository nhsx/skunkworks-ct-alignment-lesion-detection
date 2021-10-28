from ai_ct_scans.data_loading import MultiPatientAxialStreamer
import numpy as np
from skimage.filters import gabor_kernel
import cv2
from sklearn import mixture
from sklearn.cluster import MiniBatchKMeans, MeanShift
from tqdm import tqdm
from scipy.signal import medfilt2d
import pickle
from sklearn.cluster import DBSCAN
import os
import ai_ct_scans.dino.vision_transformer as vits
from PIL import Image
from torchvision import transforms as pth_transforms
from torch import nn
import torch

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = "cpu"
import matplotlib.pyplot as plt

plt.ion()


class HierarchicalMeanShift:
    """A clustering algorithm that performs MeanShiftWithProbs, then performs a second MeanShiftWithProbs for each
    class discovered in the first MeanShiftWithProbs by separating the data points by found class and finding
    probabilities of each point in that class belonging to that class, and training a new MeanShiftWithProbs clusterer
    on those probabilities for each class.
    The second order MeanShiftWithProbs clusterers do not see training points outside the first order class to which
    they are fitting.

    """

    def __init__(self):
        self.base_clusterer = MeanShiftWithProbs()
        self.second_level_clusterers = {}

    def fit(self, samples):
        """Trains the base_clusterer and second_level_clusterers

        Args:
            samples (np.ndarray): N samples by M dimensions dataset
        """
        self.base_clusterer.fit(samples)
        predictions = self.base_clusterer.predict(samples)
        probs = self.base_clusterer.predict_proba(samples)
        print("Training sub-clusterers")
        for cluster_label in tqdm(range(len(self.base_clusterer.cluster_centers_))):
            mask = predictions == cluster_label
            curr_samples = samples[mask]
            curr_probs = probs[:, cluster_label][mask].reshape(-1, 1)
            if len(curr_samples) > 0:
                self.second_level_clusterers[cluster_label] = MeanShiftWithProbs()
                self.second_level_clusterers[cluster_label].fit(curr_probs)

    def predict(self, samples):
        """Predicts the first order class of an array of samples

        Args:
            samples (np.ndarray): N samples by M dimensions

        Returns:
            (np.ndarray of ints): N predictions

        """
        return self.base_clusterer.predict(samples)

    def predict_proba(self, samples, cluster_label=None):
        """Predict probability of samples' membership to particular clusters, using first order clusterer only.
            Following the naming convention of sklearn's other clusterer's, having predict_proba, to enable integration
            with other methods

        Args:
            samples (np.ndarray): N by M data points
            cluster_label (int or None): A cluster for which to predict the probability of each data point in sample's
            membership. If None, return the probabilities for each class

        Returns:
            (np.ndarray): The probabilities of membership for each data point. If cluster_label was None, this will be
            shape (N, [number of clusters known by clusterer]), if cluster_label was an index of a known class, it will
            be shape (N,)

        """
        return self.base_clusterer.predict_proba(
            samples=samples, cluster_label=cluster_label
        )

    def predict_full(self, samples):
        """Predicts the class according to second order clusterers. First use the first order clusterer to section
        the samples down to those that should feed into each sub-clusterer, then label with the sub-clusterer

        Args:
            samples (np.ndarray): N samples by M dimensions

        Returns:
            (np.ndarray of ints): N predictions

        """
        second_predictions = np.ones(len(samples)) * -1
        predictions = self.base_clusterer.predict(samples)
        probs = self.base_clusterer.predict_proba(samples)
        tot_labels = 0
        for cluster_label in range(len(self.base_clusterer.cluster_centers_)):
            mask = predictions == cluster_label
            curr_sample = probs[:, cluster_label][mask].reshape(-1, 1)
            if len(curr_sample) > 0:
                curr_predictions = self.second_level_clusterers[cluster_label].predict(
                    curr_sample
                )
                second_predictions[mask] = curr_predictions + tot_labels
            tot_labels += len(
                self.second_level_clusterers[cluster_label].cluster_centers_
            )
        return second_predictions

    def predict_secondary(self, samples, primary_label):
        """Predicts the second order class of an array of samples, within a class primary_label predicted by the first
        order clusterer. These predictions will start at class 0 and run to the number of clusters found when
        the relevant sub-clusterer was first trained, and hence will have a value offset when compared to predictions
        made by predict_full, which starts each new sub-clusterer's labels at the running total of clusters found
        by previous sub-clusterers

        Args:
            samples (np.ndarray): N samples by M dimensions
            primary_label (int): The numerical class from the first order clusterer within which you wish to return
            second order predictions

        Returns:
            (np.ndarray of ints): N predictions

        """
        probs = self.base_clusterer.predict_proba(
            samples, cluster_label=primary_label
        ).reshape(-1, 1)
        return self.second_level_clusterers[primary_label].predict(probs)

    def predict_proba_secondary(self, samples, primary_label, sub_cluster_label=None):
        """Get the probability predictions from a second order clusterer on samples

        Args:
            samples (np.ndarray): N data points by M dimensions set of samples to predict probability on
            primary_label (int): The index of the secondary clusterer associated with the label predicted by the primary
            clusterer
            sub_cluster_label (int): The index of the sub-class from the secondary clusterer for which you want to
            predict the probabilities of membership of for samples

        Returns:
            (np.ndarray): 1D set of probabilities, same length as samples

        """
        probs = self.base_clusterer.predict_proba(
            samples, cluster_label=primary_label
        ).reshape(-1, 1)
        return self.second_level_clusterers[primary_label].predict_proba(
            probs, cluster_label=sub_cluster_label
        )


class MeanShiftWithProbs(MeanShift):
    """A class for getting probability predictions of class membership using the sklearn MeanShift algorithm"""

    def __init__(self):
        super(MeanShiftWithProbs, self).__init__()
        self.deviations = []

    def predict_proba(self, samples, cluster_label=None):
        """Predict probability of samples' membership to particular clusters. Following the naming
        convention of sklearn's other clusterer's, having predict_proba, to enable integration with other methods

        Args:
            samples (np.ndarray): N by M data points
            cluster_label (int or None): A cluster for which to predict the probability of each data point in sample's
            membership. If None, return the probabilities for each class

        Returns:
            (np.ndarray): The probabilities of membership for each data point. If cluster_label was None, this will be
            shape (N, [number of clusters known by clusterer]), if cluster_label was an index of a known class, it will
            be shape (N,)

        """

        if cluster_label is None:
            out_probs = np.zeros([len(samples), len(self.deviations)])
            for i in range(len(self.deviations)):
                dist_from_centre = np.linalg.norm(
                    samples - self.cluster_centers_[i], axis=1
                )
                probs = np.exp(-dist_from_centre / (1e-7 + self.deviations[i]))
                out_probs[:, i] = probs[:]
        else:
            dist_from_centre = np.linalg.norm(
                samples - self.cluster_centers_[cluster_label], axis=1
            )
            out_probs = np.exp(
                -dist_from_centre / (1e-7 + self.deviations[cluster_label])
            )

        return out_probs

    def fit(self, samples):
        """Trains the clusterer

        Args:
            samples (np.ndarray): N samples by M dimensions dataset

        """
        super().fit(samples)
        labels = self.predict(samples)
        for cluster_label in range(len(self.cluster_centers_)):
            sub_sample = samples[labels == cluster_label]
            dists_from_centre = np.linalg.norm(
                sub_sample - self.cluster_centers_[cluster_label], axis=1
            )
            self.deviations.append(np.std(dists_from_centre))
        self.deviations = np.array(self.deviations)


class TextonSectioner:
    """Section images using textons - generate per-pixel descriptors using convolution-based filters or simple
    intensity values, use clustering algorithms to separate these descriptors into classes, and enable sectioning of
    new images using the trained clusterers.
    TextonSectioner runs through axial views of patients from the dataset, by default randomly, to generate descriptors.

    """

    def __init__(
        self,
        filter_type="intensity",
        total_samples=100000,
        samples_per_image=50,
        kernels=None,
        blur_kernel=None,
        clusterers=None,
        clusterer_titles=None,
        medfilt_kernel=None,
    ):
        """Set up the kernels needed to generate the textons and the dataset to be trained upon

        Args:
            filter_type (str or list of str): The type or types of filter to include. Currently allowed are 'intensity',
                'gabor' and 'circular_cosine'. Typical best results have been seen with only 'intensity'
            total_samples (int): Number of single pixel descriptors to build from the full dataset as a training
                dataset, selected randomly from each image until the total_samples are filled
            samples_per_image (int): Number of single pixel descriptors to extract from each image before moving
                to a new image to extract further descriptors from, until total_samples is filled
            kernels (list of np.ndarrays): A custom set of kernels with which to perform convolutions on image to
                generate texton descriptors.
            blur_kernel (tuple of 2 ints): The shape of a kernel to use cv2.blur with prior to applying convolution,
                typically for removing high frequency noise, but not typically used if only 'intensity' used in
                filter_type
            clusterers (list of suitable clustering objects): A list of clusterers, e.g.
                sklearn.cluster.BayesianGaussianMisture. Must have fit, predict, predict_proba methods, ruling out some
                sklearn clusterers which cannot predict on new data, e.g. DBSCAN
            clusterer_titles (list of str): A list of titles for each clusterer in clusterers, to help keep track when
                experimenting and plotting results
            medfilt_kernel (tuple of 2 ints): The shape of a kernel to use scipy.signal.medfilt2d with, after applying
                other kernels. This is intended to minimise edge effects when convolution kernels overlap two different
                tissue types, leading to extraneous classes to be predicted for the interface between two different
                tissues, but as 'intensity' does not suffer from these issues and led to the most promising result,
                experiments with medfilt2d filtering did not progress far
        """
        self.filter_type = filter_type
        self.medfilt_kernel = medfilt_kernel
        self.dataset = MultiPatientAxialStreamer()
        self.complex_kernels = None
        if self.filter_type == "gabor" or "gabor" in self.filter_type:
            self.num_angles = 8
            self.num_sigmas = 2
            self.num_frequencies = 2
        if (
            self.filter_type == "circular_cosine"
            or "circular_cosine" in self.filter_type
        ):
            self.num_circular_sigmas = 1
            self.num_circular_frequencies = 2
        if kernels is None:
            self.filters = self._build_filters()
        else:
            self.filters = kernels
        self.total_filters = len(self.filters)
        self.total_samples = total_samples
        self.samples_per_image = samples_per_image
        if clusterers is None:
            self.clusterer_titles = []
            self.clusterers = self._build_clusterers()
        else:
            self.clusterers = clusterers
            if clusterer_titles is None:
                self.clusterer_titles = [
                    f"Custom_{n}" for n in range(len(self.clusterers))
                ]
            else:
                self.clusterer_titles = clusterer_titles
        self.blur_kernel = blur_kernel
        self.radial_mask = None

    def _circular_cosine_filter(self, extent=7, sigma=2, frequency=1):
        """Build circularly symmetric cosine kernels that decay exponentially towards the edges

        Args:
            extent (int): The square size of the output filter
            sigma (numeric): The inverse exponential decay factor
            frequency (numeric): The frequency to apply within cosine(radius*frequency) to generate the filter

        Returns:
            (np.ndarray): A 2D filter

        """
        start = int(np.ceil(-extent / 2))
        end = int(np.ceil(extent / 2))
        x, y = np.mgrid[start:end, start:end]
        r = np.sqrt(x ** 2 + y ** 2)
        kernel = np.abs(np.cos(r * frequency) * np.exp(-((r / sigma) ** 2)))
        kernel /= abs(np.sum(kernel))
        return kernel

    def _build_filters(self):
        """Goes through the defined filter_types and builds a list of suitable filters

        Returns:
            (list of np.ndarray): List of filters to generate textons with

        """
        kernels = []
        if self.filter_type == "gabor" or "gabor" in self.filter_type:
            for angle in np.linspace(0, np.pi, self.num_angles):
                for sigma in np.linspace(1, 2, self.num_sigmas):
                    for frequency in np.linspace(0.9, 3.2, self.num_frequencies):
                        kernels.append(
                            gabor_kernel(
                                frequency, theta=angle, sigma_x=sigma, sigma_y=sigma
                            ).real
                        )
        if self.filter_type == "rand" or "rand" in self.filter_type:
            for _ in range(10):
                kernels.append(np.random.rand(3, 3))
        if (
            self.filter_type == "circular_cosine"
            or "circular_cosine" in self.filter_type
        ):
            for sigma in np.linspace(3, 5, self.num_circular_sigmas):
                for frequency in np.linspace(2, 5, self.num_circular_frequencies):
                    kernels.append(
                        self._circular_cosine_filter(
                            sigma=sigma, frequency=frequency
                        ).real
                    )

        if self.filter_type == "intensity" or "intensity" in self.filter_type:
            kernels.append(
                np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]).astype("float32")
            )
        self.complex_kernels = False
        for kernel in kernels:
            if np.iscomplex(kernel).any():
                self.complex_kernels = True
        return kernels

    def _build_clusterers(self):
        """Generates a default set of clusterers to train against texton set

        Returns:
            (list): A list of clusterer objects

        """
        clusterers = []
        for n_clusters in range(3, 30, 8):
            clusterers.append(MiniBatchKMeans(n_clusters=n_clusters, random_state=555))
            self.clusterer_titles.append(f"MiniBatch{n_clusters}")
            clusterers.append(mixture.GaussianMixture(n_components=n_clusters))
            self.clusterer_titles.append(f"GaussianM{n_clusters}")
            clusterers.append(mixture.BayesianGaussianMixture(n_components=n_clusters))
            self.clusterer_titles.append(f"BayesGaus{n_clusters}")
        clusterers.append(MeanShift())
        self.clusterer_titles.append("MeanShift")
        return clusterers

    def _optional_blur(self, im):
        """If a blur_kernel was set during initialisation, use it. Otherwise, return the input image

        Args:
            im (np.ndarray): 2D image

        Returns:
            (np.ndarray): The optionally blurred image

        """
        if self.blur_kernel is None:
            return im
        else:
            return cv2.blur(im, self.blur_kernel)

    def single_image_texton_descriptors(self, im):
        """Get the texton descriptors for each pixel in an image

        Args:
            im (np.ndarray): 2D image

        Returns:
            (3D ndarray): The texton descriptors for the image, shaped into (number of descriptors, *im.shape)

        """
        output_stack = np.zeros([self.total_filters, *im.shape])
        im = self._optional_blur(im)
        for filt_i, filter in enumerate(self.filters):
            if self.complex_kernels is True:
                real_output = cv2.filter2D(im, -1, np.real(filter))
                imag_output = cv2.filter2D(im, -1, np.imag(filter))
                output = np.sqrt(real_output ** 2 + imag_output ** 2)
            else:
                output = cv2.filter2D(im, -1, filter)
            if self.medfilt_kernel is not None:
                output = medfilt2d(output, self.medfilt_kernel)
            output_stack[filt_i] = output
        return output_stack

    def build_sample_texton_set(self, threshold=None, random=True):
        """Cycle through images from the dataset and build up a texton sample set in self.texton_sample_set

        Args:
            threshold (int): Value below which not to accept texton descriptors into the training set for. Typically
            500, to rule out air, which dominates the training set otherwise
            random (bool): Whether to select images randomly from the MultiPatientAxialStreamer or simply step through
            the first patient followed by the second patient etc

        """
        self.texton_sample_set = np.zeros([self.total_samples, self.total_filters])
        num_ims = int(np.ceil(self.total_samples / self.samples_per_image))
        print("Building texton dataset")
        for im_i in tqdm(range(num_ims)):
            im = self.dataset.stream_next(threshold=threshold, random=random)
            im = self._optional_blur(im)
            descriptors = self.single_image_texton_descriptors(im)
            curr_sample_num = min(
                self.total_samples - im_i * self.samples_per_image,
                self.samples_per_image,
            )
            if threshold is not None:
                above_thresh = (im > threshold).nonzero()
                valid_indices = list(range(len(above_thresh[0])))
                choices = np.random.choice(valid_indices, size=curr_sample_num)
                curr_rows = above_thresh[0][choices]
                curr_cols = above_thresh[1][choices]
            else:
                curr_rows = np.random.randint(0, high=im.shape[0], size=curr_sample_num)
                curr_cols = np.random.randint(0, high=im.shape[1], size=curr_sample_num)
            curr_samples = descriptors[:, curr_rows, curr_cols]
            curr_start_index = im_i * self.samples_per_image
            self.texton_sample_set[
                curr_start_index : curr_start_index + curr_sample_num
            ] = curr_samples.T

    def train_clusterers(self, clusterer_inds=None):
        """Train each clusterer in self.clusterers against the texton dataset, or a subset of clusterers. If any
        clusterer fails to train, remove it from self.clusterers

        Args:
            clusterer_inds (list of ints, optional): The indices of clusterers to train, defaults to train all of them

        """
        print("Training clusterers")
        if clusterer_inds is None:
            clusterer_inds = list(range(len(self.clusterers)))
        failed_inds = []
        for ind in tqdm(clusterer_inds):
            try:
                self.clusterers[ind].fit(self.texton_sample_set)
            except ValueError:
                failed_inds.append(ind)
        for ind in failed_inds:
            self.clusterers.pop(ind)
            print(f"Failed to fit clusterer {self.clusterer_titles[ind]}")
            self.clusterer_titles.pop(ind)

    def label_im(
        self,
        im,
        threshold=None,
        clusterer_ind=0,
        sub_structure_class_label=None,
        full_sub_structure=False,
    ):
        """Sections a new image with class labels assigned by a trained clusterer

        Args:
            im (np.ndarray): 2D image to be labelled pixel-wise
            threshold (int): A value below which to assign all image pixel classes to -1, useful for sectioning out
            air with a threshold of ~500 when clusterers have not been trained with air in the texton sample set
            clusterer_ind (int): The clusterer index in self.clusterers you wish to label the image with. Defaults to 0
            sub_structure_class_label (int or None): The class predicted by the clusterer at clusterer_ind in
            self.clusterers you wish to predict sub-class labels for - this must be used with a
            self.clusterer[clusterer_ind] that has hierarchical style labelled, i.e. has a predict_secondary method, as
            in HierarchicalMeanShift. If None, only use a first order clusterer
            full_sub_structure (bool, optional): If a  hiererachical clusterer has been selected with clusterer_ind,
            whether to return the full sub-class predictions rather than the first order predictions

        Returns:
            (np.ndarray of ints): The class predictions, same shape as im

        """
        im = self._optional_blur(im)
        textons = self.single_image_texton_descriptors(im)
        textons = textons.reshape(
            [textons.shape[0], textons.shape[1] * textons.shape[2]]
        ).T
        if sub_structure_class_label is not None:
            super_class_label_im = self.label_im(
                im, threshold=threshold, clusterer_ind=clusterer_ind
            )
            out_im = self.clusterers[clusterer_ind].predict_secondary(
                textons, primary_label=sub_structure_class_label
            )
            """
            section the sub_structure image down to only include pixels which originally belonged to its super class
            add 1 then multiply then minus 1 to deal with thresholded out pixels remaining set to -1 and class 0 pixels
            from sub-clusterer not eliminating information from the super class
            """
            out_im = out_im.reshape(*im.shape)
            out_im = (out_im + 1) * (
                super_class_label_im == sub_structure_class_label
            ) - 1
        elif full_sub_structure is True:
            out_im = self.clusterers[clusterer_ind].predict_full(textons)
            out_im = out_im.reshape(*im.shape)
        else:
            out_im = self.clusterers[clusterer_ind].predict(textons)
            out_im = out_im.reshape(*im.shape)
        if threshold is not None:
            out_im[im < threshold] = -1
        return out_im

    def probabilities_im(
        self,
        im,
        threshold=None,
        clusterer_ind=0,
        cluster_label=None,
        return_sub_structure=False,
        sub_structure_class_label=None,
    ):
        """Get the probabilities that each pixel in an image belong to a particular class predicted by a clusterer, as
        well as getting the class predictions image itself

        Args:
            im (np.ndarray): 2D image to be labelled pixel-wise
            threshold (int): A value below which to assign all image pixel classes to -1, useful for sectioning out
            air with a threshold of ~500 when clusterers have not been trained with air in the texton sample set
            clusterer_ind (int): The clusterer index in self.clusterers you wish to label the image with. Defaults to 0
            cluster_label (int): The class within which to predict probabilities for
            return_sub_structure (bool): Whether to use a secondary clusterer to predict the probabilities, e.g. from
            HierarchicalMeanShift
            sub_structure_class_label (int): If return_sub_structure is True, the sub-class label to predict
            probabilities for

        Returns:
            (tuple of np.ndarrays): First element: The class predictions for each pixel, second element: the
            probabilities image

        """
        im = self._optional_blur(im)
        textons = self.single_image_texton_descriptors(im)
        textons = textons.reshape(
            [textons.shape[0], textons.shape[1] * textons.shape[2]]
        ).T
        out_im = self.clusterers[clusterer_ind].predict(textons)
        if cluster_label is not None:

            if return_sub_structure is True:
                probabilities = self.clusterers[clusterer_ind].predict_proba_secondary(
                    textons,
                    primary_label=cluster_label,
                    sub_cluster_label=sub_structure_class_label,
                )
            else:
                probabilities = self.clusterers[clusterer_ind].predict_proba(
                    textons, cluster_label=cluster_label
                )
            probabilities = probabilities.reshape(*im.shape)
        else:
            probabilities = self.clusterers[clusterer_ind].predict_proba(textons).T
            probabilities = probabilities.reshape(-1, *im.shape)
        out_im = out_im.reshape(*im.shape)
        if threshold is not None:
            out_im[im < threshold] = -1
        return out_im, probabilities

    def save(self, out_path):
        """Save the TextonSectioner using pickle. Only the minimal set of clusterers, clusterer_titles and filters that
        are required to load and produce new predictions on images are saved.

        Args:
            out_path (pathlib Path to a .pkl file): Where to save the TextonSectioner

        """
        with open(out_path, "wb") as file:
            state = (self.clusterers, self.clusterer_titles, self.filters)
            pickle.dump(state, file)

    def load(self, load_path):
        """Reload a TextonSectioner using pickle

        Args:
            load_path (pathlib Path): Path to a pickled TextonSectioner

        """
        with open(load_path, "rb") as file:
            state = pickle.load(file)
            self.clusterers, self.clusterer_titles, self.filters = state


class EllipseFitter:
    def __init__(
        self,
        min_area_ratio=0.75,
        min_eccentricity=0.5,
        min_ellipse_long_axis=0.0,
        max_ellipse_long_axis=np.inf,
        max_ellipse_contour_centre_dist=10,
        min_area=25,
        max_area=10000,
    ):
        """A class for fitting ellipses around regions of pre-sectioned images

        Args:
            min_area_ratio (float): The minimum fraction of a fitted ellipse's that the contour used to find the ellipse
            must encompass to be accepted
            min_eccentricity (float): The minimum short axis to long axis ratio for an ellipse to be accepted
            min_ellipse_long_axis (int): The maximum pixel length of an ellipse long axis to be accepted
            max_ellipse_long_axis (int): The minimum pixel length of an ellipse long axis to be accepted
            max_ellipse_contour_centre_dist (float): The maximum distance between an ellipse centre and the centre of
            mass of the contour used to find it to be accepted as an ellipse
            min_area (float): The minimum area of an ellipse to be accepted as an ellipse
            max_area (float): The maximum area of an ellipse to be accepted as an ellipse
        """
        self.min_area_ratio = min_area_ratio
        self.min_area = min_area
        self.max_area = max_area
        self.min_eccentricity = min_eccentricity
        self.min_ellipse_long_axis = min_ellipse_long_axis
        self.max_ellipse_long_axis = max_ellipse_long_axis
        self.max_ellipse_contour_centre_dist = max_ellipse_contour_centre_dist

    def fit_ellipses(self, image, background_val=0):
        """Find valid ellipses in a pre-sectioned 2D image. Pre-sectioned here meaning that pixels in an original image
        have been replaced by class labels, such that nearby pixels are likely to share properties and therefore have
        been set to the same class

        Args:
            image (np.ndarray): The 2D image within which to find ellipses
            background_val (int): A valid uint8 number to be considered as background and skipped, default 0

        Returns:
            (list of tuples of floats): A list of found ellipses, each as returned by cv2.fitEllipse
        """
        # section to each unique value in the image (except background) and store found contours
        im = image.astype("uint8")
        contours = []
        contour_classes = []
        for val in np.unique(im):
            if val == background_val:
                continue
            curr_im = im == val
            curr_contours, _ = cv2.findContours(
                curr_im.astype("uint8"), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
            )
            contours += curr_contours
            contour_classes += [val] * len(curr_contours)

        ellipses = []
        ellipse_classes = []
        for contour, contour_class in zip(contours, contour_classes):
            if len(contour) < 5:
                # need at least 5 points to fit an ellipse
                continue

            # get moment of contour, to be used for centre of mass check later as well as whether it has any area at all
            moment = cv2.moments(contour)
            if moment["m00"] == 0:
                # if mass of contour is zero then it's not enclosing anything - no area
                continue

            # do a quick check on whether the contour with a convex hull will be within size allowances
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            if hull_area < self.min_area:
                continue
            if hull_area > self.max_area:
                continue

            # fit the actual ellipse
            ellipse = cv2.fitEllipse(contour)

            if ellipse[1][0] / ellipse[1][1] < self.min_eccentricity:
                continue
            if ellipse[1][1] < self.min_ellipse_long_axis:
                continue
            if ellipse[1][1] > self.max_ellipse_long_axis:
                continue

            com = np.array(
                [moment[component] / moment["m00"] for component in ["m10", "m01"]]
            )
            dist = np.linalg.norm(com - np.array(ellipse[0]))
            if dist > self.max_ellipse_contour_centre_dist:
                continue

            contour_area = cv2.contourArea(contour)
            ellipse_area = ellipse[1][0] * ellipse[1][1] * np.pi / 4
            if not (contour_area / ellipse_area) > self.min_area_ratio:
                continue
            ellipses.append(ellipse)
            ellipse_classes.append(contour_class)
        return ellipses, ellipse_classes


class CTEllipsoidFitter:
    """Class for fitting ellipses within 3D CT scans"""

    def __init__(
        self,
        min_area_ratio=0.75,
        min_eccentricity=0.5,
        min_ellipse_long_axis=0.0,
        max_ellipse_long_axis=np.inf,
        max_ellipse_contour_centre_dist=10,
        min_area=25,
        max_area=25000,
    ):
        self.ellipse_fitter = EllipseFitter(
            min_area_ratio=min_area_ratio,
            min_eccentricity=min_eccentricity,
            min_ellipse_long_axis=min_ellipse_long_axis,
            max_ellipse_long_axis=max_ellipse_long_axis,
            max_ellipse_contour_centre_dist=max_ellipse_contour_centre_dist,
            min_area=min_area,
            max_area=max_area,
        )

    def draw_ellipses_2d(self, image):
        """Fit ellipses within an image and then draw them onto a blank image of the same dimensions

        Args:
            image (np.ndarray): 2D grayscale image. Internally converted to uint8 values, so pixel values should round
            to distinct integers between 1-255 (0 treated as background) for ellipses to be extracted as expected

        Returns:
            (np.ndarray): 2D image with ellipses drawn on

        """
        ellipses, ellipse_classes = self.ellipse_fitter.fit_ellipses(image)
        out_im = np.zeros_like(image, dtype="uint8")
        for ellipse in ellipses:
            cv2.ellipse(out_im, ellipse, color=1, thickness=1)

        return out_im, ellipses, ellipse_classes

    def draw_ellipsoid_walls(
        self,
        arr,
        sectioner=None,
        sectioner_kwargs=None,
        filterer=None,
        filter_kernel=None,
        return_sectioned=False,
    ):
        """Runs through a 3D array of scan data in each dimension, sectioning and then drawing ellipses around any 2D
        elliptical structures found. By doing this in each axis, ellipsoidal shells are built up, which can guide
        to the location of ellipsoid_volume lesions

        Args:
            arr (np.ndarray): 3D CT scan data
            sectioner (TextonSectioner): A sectioning object with a .label_im method, to pixel-wise label an image
            sectioner_kwargs (dict): Kwargs for the instantiation of a sectioner object
            filterer (method): A filtering method to apply after sectioning, typically scipy.signal.medfilt2d, which can
            round the edges of sectioned tissue boundaries.
            filter_kernel (tuple of ints): Shape of the kernel to use with filterer
            return_sectioned (bool): Whether to return the sectioned 3D scan as well as the ellipsoidal view

        Returns:
            (tuple of ndarrays or ndarray): if return_sectioned=False, only return a 3D ndarray with 1s, 2s or 3s
            wherever ellipse edges were detected in each 2D slice. A value of 1 means an ellipse edge was only detected
            at that pixel in a single axis, 2 in 2 axes, 3 in 3 axes. If return_sectioned=True, also return a 3D ndarray
            of the sectioned CT scan

        """
        ellipsoid_volume = np.zeros_like(arr, dtype="uint8")
        all_ellipses = []
        if sectioner is not None:
            sectioned = np.copy(ellipsoid_volume)
        for axis in list(range(arr.ndim)):
            for im_i in tqdm(range(arr.shape[axis])):
                if axis == 0:
                    s_ = np.s_[im_i, :, :]
                if axis == 1:
                    s_ = np.s_[:, im_i, :]
                if axis == 2:
                    s_ = np.s_[:, :, im_i]

                if sectioner is not None:
                    if axis == 0:
                        curr_im = arr[s_]
                        curr_im = (
                            sectioner.label_im(curr_im, **sectioner_kwargs) + 1
                        ).astype("uint8")
                        if filterer is not None:
                            curr_im = filterer(curr_im, filter_kernel)
                        sectioned[s_] = curr_im
                    elif axis > 0:
                        curr_im = sectioned[s_]
                else:
                    curr_im = arr[s_]

                ellipse_im, ellipses, ellipse_classes = self.draw_ellipses_2d(curr_im)
                if len(ellipses) > 0:
                    all_ellipses += self._ellipses_to_rich_information(
                        ellipses, axis, im_i, ellipse_classes
                    )
                ellipsoid_volume[s_] += ellipse_im

        accepted_ellipsoids = self.find_ellipsoids(all_ellipses)

        if return_sectioned:
            return ellipsoid_volume, accepted_ellipsoids, sectioned
        return ellipsoid_volume, accepted_ellipsoids

    def _ellipses_to_rich_information(
        self, ellipses, axis, axis_layer, ellipse_classes
    ):
        """Amends a list of ellipse-like objects (ala the output of cv2.findEllipse) to 3D (axial, coronal, sagittal)
        coordinates, as well as the original axis the ellipse was detected in, the class of pixel as detected by a
        tissue sectioner, and the ellipse's area, for each ellipse in the list. Every ellipse is expected to have been
        found in a single image, such that they all come from a single axis and a single layer along that axis

        Args:
            ellipses (list of tuple): Each element is a definition of an ellipse, as from cv2.findEllipse
            axis (int): The axis from which all ellipses in ellipses were found, i.e. 0 if they were found when viewing
            in the axial plane, 1 for coronal, 2 for sagittal
            axis_layer (int): The slice index at which the image in which the ellipses were found, i.e. 123 if the
            ellipses were found in the 123rd slice along axis axis
            ellipse_classes (list of ints): The class of pixel associated with each ellipse. If a sectioner was used,
            this will be the tissue class with a +1 offset, otherwise it will be the raw pixel value

        Returns:
            (list of tuples): Each element is a 'rich ellipse' with 3D coordinates at the 0th element, long and short
            axes lengths at the 1st element, angle of rotation at the 2nd, originating axis at the 3rd, sectioned class
            of pixels belonging to the ellipse at 4th, area at 5th

        """
        out_ellipses = []
        for ellipse, ellipse_class in zip(ellipses, ellipse_classes):
            if axis == 0:
                new_coords = (axis_layer, ellipse[0][1], ellipse[0][0])
            elif axis == 1:
                new_coords = (ellipse[0][1], axis_layer, ellipse[0][0])
            elif axis == 2:
                new_coords = (ellipse[0][1], ellipse[0][0], axis_layer)
            ellipse_area = ellipse[1][0] * ellipse[1][1] * np.pi / 4
            new_ellipse = (
                new_coords,
                ellipse[1],
                ellipse[2],
                axis,
                ellipse_class,
                ellipse_area,
            )
            out_ellipses.append(new_ellipse)

        return out_ellipses

    def _rich_ellipse_axial_area_info(self, rich_ellipses):
        """Generate per-axis area info for each ellipse in rich_ellipses

        Args:
            rich_ellipses (list of tuples): A list of rich ellipses, similar to the output of
                self._ellipses_to_rich_information. At minimum must have the axis along which the ellipse
                was found at the 3rd element of each entry, and the area of the ellipse at the 5th element of
                each entry

        Returns:
            (dict): Dictionary with keys 0, 1, 2 for axes, with sub-dictionaries of area information for the ellipses
            found along those axes, and a 'min_area_ratio' to record the minimum ratio of area found along all three
            axes

        """
        out_info = {}
        min_area_ratio = np.inf
        for axis in [0, 1, 2]:
            curr_ellipse_areas = [
                ellipse[5] for ellipse in rich_ellipses if ellipse[3] == axis
            ]
            if len(curr_ellipse_areas) > 0:
                low_area, high_area = np.percentile(curr_ellipse_areas, q=[10, 90])
                area_ratio = low_area / high_area
                # get a volume in pixels
                volume = np.sum(curr_ellipse_areas)
                if area_ratio < min_area_ratio:
                    min_area_ratio = area_ratio
                curr_dict = {
                    "min_area": np.min(curr_ellipse_areas),
                    "max_area": np.max(curr_ellipse_areas),
                    "volume": volume,
                    "area_ratio": area_ratio,
                    "num_ellipses": len(curr_ellipse_areas),
                }
            else:
                curr_dict = {
                    "min_area": np.nan,
                    "max_area": np.nan,
                    "volume": np.nan,
                    "area_ratio": np.nan,
                    "num_ellipses": 0,
                }
            out_info[axis] = curr_dict
        out_info["min_area_ratio"] = min_area_ratio
        return out_info

    def find_ellipsoids(self, rich_ellipses):
        """Find ellipsoids within a list of 2D ellipses, where the centres of the 2D ellipses in 3D space must be close
        together and of the same underlying tissue class

        Args:
            rich_ellipses (list of tuples): A list of rich ellipses, similar to the output of
                self._ellipses_to_rich_information.

        Returns:
            (list of dicts): A list of all ellipsoids found, each one a dict with a 'centre': 3D ndarray,
            'max_position': 1D ndarray length 3 (axial, coronal, sagittal max positions of a bounding box),
            'min_position': 1D ndarray length 3, minimum bounding box positions,
            'volume': ellipsoid volume in pixels,
            'axis_ellipses_count': list of ints length 3, the number of ellipses that contributed to the detection on
             an ellipsoid in each axis,
             'class': int, the numerical tissue class assigned to the ellipse, expected to be offset by one compared to
             the output of a TextonSectioner.label_im output, as -1 is treated as background by the sectioner and 0 is
             treated as background in the ellipse fitter, so an offset will have been applied to aid ellipse detection

        """
        # TODO refactor this method into smaller, more thoroughly tested sub-methods
        centres = []
        classes = []
        for ellipse in rich_ellipses:
            centres.append(ellipse[0])
            classes.append(ellipse[4])
        centres = np.stack(centres, axis=0)
        classes = np.array(classes)
        centres_by_classes = []
        unique_classes = np.unique(classes)
        ellipses_by_classes = []
        for val in unique_classes:
            curr_class_mask = classes == val
            centres_by_classes.append(centres[curr_class_mask])
            ellipses_by_classes.append(
                [ellipse for ellipse in rich_ellipses if ellipse[4] == val]
            )
        np.random.rand(555)
        clusterer = DBSCAN(eps=2.0)
        clusters_by_class = {}
        ellipse_clusters_by_class = {}
        for centres_by_class, ellipses_by_class, unique_class in zip(
            centres_by_classes, ellipses_by_classes, unique_classes
        ):
            clusterer.fit(centres_by_class)
            clusters_by_class[unique_class] = {}
            ellipse_clusters_by_class[unique_class] = {}
            for cluster_index in np.unique(clusterer.labels_):
                if cluster_index == -1:
                    # ignore points that weren't clustered
                    continue
                cluster_mask = clusterer.labels_ == cluster_index
                cluster_indices = cluster_mask.nonzero()[0]
                clusters_by_class[unique_class][cluster_index] = centres_by_class[
                    cluster_mask
                ]
                ellipse_clusters_by_class[unique_class][cluster_index] = [
                    ellipses_by_class[i]
                    for i in range(len(ellipses_by_class))
                    if i in cluster_indices
                ]
        ellipsoids = []
        for class_key in clusters_by_class.keys():
            for cluster_key in clusters_by_class[class_key].keys():
                cluster_centre = np.mean(
                    clusters_by_class[class_key][cluster_key], axis=0
                )
                cluster_max = np.max(clusters_by_class[class_key][cluster_key], axis=0)
                cluster_min = np.min(clusters_by_class[class_key][cluster_key], axis=0)
                cluster_size_per_axis = cluster_max - cluster_min

                axis_area_info = self._rich_ellipse_axial_area_info(
                    ellipse_clusters_by_class[class_key][cluster_key]
                )
                max_size = np.max(cluster_size_per_axis)
                if max_size > 0:
                    eccentricity_by_cluster_centres = (
                        np.median(cluster_size_per_axis) / max_size
                    )
                else:
                    eccentricity_by_cluster_centres = 0.0
                growth_by_axial_areas = np.sqrt(axis_area_info["min_area_ratio"])
                """
                Rule out clusters that only appear for several layers in a single axis and do not grow/shrink within
                that axis by at least a factor of 0.5, i.e. rule out cylinders
                """
                ellipsoid_check = (
                    eccentricity_by_cluster_centres
                    > self.ellipse_fitter.min_eccentricity
                ) or (growth_by_axial_areas > 0.5)
                if not (max_size > 0 and ellipsoid_check):
                    # skip the ellipsoid if it doesn't fit criteria
                    continue

                cluster_volume = np.nanmax(
                    [axis_area_info[i]["volume"] for i in range(3)]
                )
                cluster_class = class_key
                axis_ellipses_count = [
                    axis_area_info[i]["num_ellipses"] for i in [0, 1, 2]
                ]
                curr_dict = {
                    "centre": cluster_centre,
                    "max_position": cluster_max,
                    "min_position": cluster_min,
                    "volume": cluster_volume,
                    "axis_ellipses_count": axis_ellipses_count,
                    "class": cluster_class,
                    "contributing_ellipses": ellipse_clusters_by_class[class_key][
                        cluster_key
                    ],
                }
                ellipsoids.append(curr_dict)

        return ellipsoids


class DinoSectioner(TextonSectioner):
    """Class for using DINO-trained models to section images. Much of the code is refactored from the DINO repository,
    itself stored in ai_ct_scans.dino.

    """

    def __init__(self, max_thresh=4628.0, total_samples=5000, samples_per_image=500):
        self.model = None
        self.image_to_tensor_transform = None
        self.patch_size = None
        self.max_thresh = max_thresh
        self.dataset = MultiPatientAxialStreamer()
        self.total_samples = total_samples
        self.samples_per_image = samples_per_image
        self.blur_kernel = None
        self.clusterers = [HierarchicalMeanShift()]
        self.clusterer_titles = ["HierarchicalMeanShift"]

    def load_dino_model(
        self,
        arch="vit_tiny",
        patch_size=16,
        pretrained_weights="",
        checkpoint_key="teacher",
    ):
        model = vits.__dict__[arch](patch_size=patch_size, num_classes=0)
        for p in model.parameters():
            p.requires_grad = False
        model.eval()
        model.to(dev)
        self.patch_size = patch_size

        if os.path.isfile(pretrained_weights):
            state_dict = torch.load(pretrained_weights, map_location="cpu")
            if checkpoint_key is not None and checkpoint_key in state_dict:
                print(f"Take key {checkpoint_key} in provided checkpoint dict")
                state_dict = state_dict[checkpoint_key]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = model.load_state_dict(state_dict, strict=False)
            print(
                "Pretrained weights found at {} and loaded with msg: {}".format(
                    pretrained_weights, msg
                )
            )
        else:
            print(
                "Please use the `--pretrained_weights` argument to indicate the path of the checkpoint to evaluate."
            )
            url = None
            if arch == "vit_small" and patch_size == 16:
                url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
            elif arch == "vit_small" and patch_size == 8:
                url = (
                    "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
                )
            elif arch == "vit_base" and patch_size == 16:
                url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
            elif arch == "vit_base" and patch_size == 8:
                url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
            if url is not None:
                print(
                    "Since no pretrained weights have been provided, we load the reference pretrained DINO weights."
                )
                state_dict = torch.hub.load_state_dict_from_url(
                    url="https://dl.fbaipublicfiles.com/dino/" + url
                )
                model.load_state_dict(state_dict, strict=True)
            else:
                print(
                    "There is no reference weights available for this model => We use random weights."
                )

        self.model = model
        self.total_filters = self.model.blocks[-1].attn.num_heads

    def _ct_slice_to_uint8(self, image):
        """Normalises an image to the uint8 range [0, 255], while also applying a threshold maximum. If the image is
        already uint8 or no self.max_thresh is set, just return the image
        maximum
        threshold

        Args:
            image (ndarray): a 2D image

        Returns:
            (ndarray): a 2D uint8 image

        """
        if self.max_thresh is None or image.dtype == "uint8":
            return image
        image = image.astype(float)
        image[image > self.max_thresh] = self.max_thresh
        image /= self.max_thresh
        image *= 255
        return image.astype("uint8")

    def single_image_texton_descriptors(self, image, threshold=None):
        """Get the texton descriptors for a single image. This is largely a refactoring of code from the DINO
        repository, which can be seen in ai_ct_scans/dino/visualize_attention

        Args:
            image (ndarray):
            threshold:

        Returns:

        """
        if self.image_to_tensor_transform is None:
            self.image_to_tensor_transform = pth_transforms.Compose(
                [
                    pth_transforms.Resize(image.shape[1:]),
                    pth_transforms.ToTensor(),
                    pth_transforms.Normalize(
                        (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                    ),
                ]
            )

        img = self._ct_slice_to_uint8(image)
        img = Image.fromarray(img).convert("RGB")
        img = self.image_to_tensor_transform(img)

        # make the image divisible by the patch size
        w, h = (
            img.shape[1] - img.shape[1] % self.patch_size,
            img.shape[2] - img.shape[2] % self.patch_size,
        )
        img = img[:, :w, :h].unsqueeze(0)

        w_featmap = img.shape[-2] // self.patch_size
        h_featmap = img.shape[-1] // self.patch_size

        attentions = self.model.get_last_selfattention(img.to(dev))

        nh = attentions.shape[1]  # number of head

        # we keep only the output patch attention
        attentions = attentions[0, :, 0, 1:].reshape(nh, -1)

        if threshold is not None:
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(attentions)
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for head in range(nh):
                th_attn[head] = th_attn[head][idx2[head]]
            th_attn = th_attn.reshape(nh, w_featmap, h_featmap).float()
            # interpolate
            th_attn = (
                nn.functional.interpolate(
                    th_attn.unsqueeze(0), scale_factor=self.patch_size, mode="nearest"
                )[0]
                .cpu()
                .numpy()
            )

        attentions = attentions.reshape(nh, w_featmap, h_featmap)
        # altered interpolation from DINO original code, which upscaled by patch_size, to instead upscale to original
        # image input size
        attentions = (
            nn.functional.interpolate(
                attentions.unsqueeze(0), size=image.shape, mode="nearest"
            )[0]
            .cpu()
            .numpy()
        )
        return attentions

    def save(self, out_path):
        """Save the TextonSectioner using pickle. Only the minimal set of clusterers, clusterer_titles and filters that
        are required to load and produce new predictions on images are saved.

        Args:
            out_path (pathlib Path to a .pkl file): Where to save the TextonSectioner

        """
        with open(out_path, "wb") as file:
            state = self.clusterers
            pickle.dump(state, file)

    def load(self, load_path):
        """Reload a TextonSectioner using pickle

        Args:
            load_path (pathlib Path): Path to a pickled TextonSectioner

        """
        with open(load_path, "rb") as file:
            state = pickle.load(file)
            self.clusterers = state
