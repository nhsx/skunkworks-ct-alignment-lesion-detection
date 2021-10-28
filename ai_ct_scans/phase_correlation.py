from skimage.registration import phase_cross_correlation
from skimage.transform import SimilarityTransform, warp
import numpy as np
from ai_ct_scans import phase_correlation_image_processing


def find_shift_via_phase_correlation_2d(image_1, image_2):
    """Get the shift from 2D image_1 to 2D image_2 via phase correlation

    Args:
        image_1 (np.ndarray): An image
        image_2 (np.ndarray): An image, expected to be similar to image_1 with a translation being the major
        transform
        between them

    Returns:
        (np.ndarray): The [col, row] translation between image_1 and image_2

    """
    shift = phase_cross_correlation(image_1, image_2, return_error=False)
    # reverse 0 and 1 elements to get into (x, y) translation as SimilarityTransform likes
    shift[0], shift[1] = shift[1], shift[0]
    return shift


def shift_image_2d(image, shift):
    """Shifts image by shift via skimage SimilarityTransform

    Args:
        image (np.ndarray): An image
        shift (np.ndarray): Two ints

    Returns:
        (np.ndarray): An image

    """
    transform = SimilarityTransform(translation=shift)
    return warp(image, transform)


def align_via_phase_correlation_2d(image_1, image_2):
    """Aligns 2D image_2 to 2D image_1 via phase correlation

    Args:
        image_1 (np.ndarray): An image
        image_2 (np.ndarray): An image, expected to be similar to image_1 with a translation being the major transform
        between them

    Returns:
        (np.ndarray): image_2 after shifting to align with image_1

    """
    shift = find_shift_via_phase_correlation_2d(image_1, image_2)
    return shift_image_2d(image_2, -shift)


def shift_via_phase_correlation_nd(
    images,
    apply_lmr=True,
    apply_zero_crossings=True,
    lmr_filter_type=None,
    lmr_radius=3,
    zero_crossings_thresh="auto",
):
    """Gets the shift between the zeroth image and all images (including zeroth) in a list of n-dimensional images
    via phase correlation

    Args:
        lmr_filter_type: (None or ndarray): The filter to apply to find the local mean, typically a shape centred at
        [0, 0]
        apply_zero_crossings (bool): whether to mask images to points where the images cross zero before finding shift,
        zero_crossings_thresh (float or 'auto'): The amount by which adjacent pixel values must differ while crossing 0
        to be counted as a crossing point. If 'auto', chooses the bin edge value prior to the maximum counts in the
        smoothed histogram of pixel values
        lmr_radius (float): if lmr_filter_type is none, an n dimensional circle of radius lmr_radius is used as the
        filter lmr_filter_type (None or ndarray): The filter to apply to find the local mean, typically a shape centred
        at [0, 0] apply_zero_crossings (bool): whether to mask images to points where the images cross zero before
        finding shift, effectively a weighted edge detection step
        apply_lmr (bool): whether to apply local mean removal before optionally appling zero crossings and then shift
        detection - recommended to use this if using apply_zero_crossings as it should create more zero crossing points
        images (list of ndarrays): The set of images to discover shifts relative to the zeroth image for

    Returns:
        (list of tuples): The [(row, col, depth, dimension_4, ...)_0, ...] shift from images[0] to overlay on
        images[n]

    """

    if apply_lmr is True:
        images = [
            phase_correlation_image_processing.lmr(
                im, filter_type=lmr_filter_type, radius=lmr_radius
            )
            for im in images
        ]
    if apply_zero_crossings is True:
        # multiply the image by points where it crosses zero, effectively a weighted zero crossings
        images = [
            im
            * phase_correlation_image_processing.zero_crossings(
                im, thresh=zero_crossings_thresh
            )
            for im in images
        ]

    # pad images if they are not all the same size
    max_shape = phase_correlation_image_processing.max_shape_from_image_list(images)
    for im_i, im in enumerate(images):
        if im.shape != max_shape:
            # replace the image at the index with a padded version
            images[im_i] = phase_correlation_image_processing.pad_nd(im, max_shape)

    im_shape = np.array(images[0].shape)
    fft_0 = np.fft.fftn(images[0] - images[0].mean())
    # first shift will always be zero
    shifts = [np.zeros(len(images[0].shape))]
    for im in images[1:]:
        curr_fft_conj = (np.fft.fftn(im - im.mean())).conj()
        fft_multiplied = fft_0 * curr_fft_conj
        # get normalised power spectrum, dealing with any regions below machine precision
        power_spectrum = np.copy(fft_multiplied)
        power_spectrum[np.abs(power_spectrum) > 1e-10] /= np.abs(
            power_spectrum[np.abs(power_spectrum) > 1e-10]
        )
        # power_spectrum = fft_multiplied / np.abs(fft_multiplied)
        phase_corr = np.fft.ifftn(power_spectrum).real
        peak_pos = np.array(np.unravel_index(np.argmax(phase_corr), images[0].shape))
        curr_shift = []
        for pos_el, im_shape_el in zip(peak_pos, im_shape):
            if pos_el > im_shape_el / 2:
                curr_shift.append(im_shape_el - pos_el)
            else:
                curr_shift.append(-pos_el)
        shifts.append(np.array(curr_shift))
    return shifts


def shifts_via_local_region(
    images,
    local_coords,
    region_widths,
    apply_lmr=True,
    apply_zero_crossings=True,
    lmr_filter_type=None,
    lmr_radius=None,
    zero_crossings_thresh="auto",
):
    """Find the shifts between images using a reduced view extracted around the same coordinates in each image,
    optionally pre-processing via local mean removal and zero crossings before applying phase correlation

    Args:
        images (list of ndarrays): The set of images to discover shifts relative to the zeroth image for
        local_coords (list of ints): The coordinates of the central point of interest in both images
        region_widths (tuple of tuples of ints or tuple of ints): The spacing either side of the local coordinate to
        extract a slice of the images with. Can either be a single int for each dimension, i.e. take a symmetric
        spacing, or a tuple of tuples of ints where each sub-tuple is length 2, i.e. a different spacing either side of
        local coordinate
        apply_lmr (bool): whether to apply local mean removal before optionally appling zero crossings and then shift
        apply_zero_crossings (bool): whether to mask images to points where the images cross zero before finding shift,
        effectively a weighted edge detection step
        lmr_filter_type (None or ndarray): The filter to apply to find the local mean, typically a shape centred at
        [0, 0]
        lmr_radius (float): if lmr_filter_type is none, an n dimensional circle of radius lmr_radius is used as the
        filter
        zero_crossings_thresh (float or 'auto'): The amount by which adjacent pixel values must differ while crossing 0
        to be counted as a crossing

    Returns:
        (list of ndarrays): A vector shift for each each image in images relative to the zeroth image (including
        self-shift at the zeroth output position)

    """
    # want the +- delta pixels to be allowed to be defined separately or as a single value for each dimension
    region_widths = [
        region_width
        if (isinstance(region_width, list) and len(region_width) == 2)
        else [region_width, region_width]
        for region_width in region_widths
    ]
    slices = tuple(
        np.s_[local_coord - region_width[0] : local_coord + region_width[1]]
        for local_coord, region_width in zip(local_coords, region_widths)
    )
    local_regions = [np.copy(im[slices]) for im in images]
    if apply_lmr is True:
        local_regions = [
            phase_correlation_image_processing.lmr(
                local_region, filter_type=lmr_filter_type, radius=lmr_radius
            )
            for local_region in local_regions
        ]
    if apply_zero_crossings is True:
        local_regions = [
            local_region
            * phase_correlation_image_processing.zero_crossings(
                local_region, thresh=zero_crossings_thresh
            )
            for local_region in local_regions
        ]
    shifts = shift_via_phase_correlation_nd(
        local_regions, apply_lmr=False, apply_zero_crossings=False
    )
    return shifts


def shift_nd(image, shift):
    """Shifts an image across an arbitrary number of dimensions by a vector shift defining the shift in each dimension

    Args:
        image (ndarray): an image
        shift (ndarray): 1D vector of ints by which to shift the image

    Returns:
        (ndarray): The original image, shifted by the specified amount, with border regions filled in with zeros
    """
    image = np.roll(image, shift, axis=tuple(range(len(image.shape))))
    for shift_i, shift_element in enumerate(shift):
        slices = []
        for shape_i in list(range(len(image.shape))):
            if shift_i == shape_i:
                if shift_element < 0:
                    slices.append(np.s_[shift_element:])
                elif shift_element > 0:
                    slices.append(np.s_[:shift_element])
                else:
                    slices.append(np.s_[0:0])
            else:
                slices.append(np.s_[:])
        slices = tuple(slices)
        image[slices] = 0
    return image
