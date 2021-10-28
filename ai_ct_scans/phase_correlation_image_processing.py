import numpy as np
from kneed import KneeLocator
from scipy import ndimage


def circle(array_size, radius):
    """Get a solid circle of True on a False background, centred at [0, 0]

    Args:
        array_size (list of ints): The [number of rows, number of columns] to generate the circle on
        radius (float): The radius of the circle

    Returns:
        (ndarray): A boolean solid circle of True on a False background centred at [0, 0]

    """
    y_start = np.ceil(-array_size[0] / 2)
    y_end = np.ceil(array_size[0] / 2)
    x_start = np.ceil(-array_size[1] / 2)
    x_end = np.ceil(array_size[1] / 2)
    y, x = np.mgrid[y_start:y_end, x_start:x_end]
    return np.fft.ifftshift(x ** 2 + y ** 2 <= radius ** 2)


def sphere(array_size, radius):
    """Get a solid circle of True on a False background, centred at [0, 0]

    Args:
        array_size (list of ints): The [number of layers, number of rows, number of columns] to generate the sphere on
        radius (float): The radius of the sphere

    Returns:
        (ndarray): A boolean solid circle of True on a False background centred at [0, 0]

    """
    y_start = np.ceil(-array_size[1] / 2)
    y_end = np.ceil(array_size[1] / 2)
    x_start = np.ceil(-array_size[2] / 2)
    x_end = np.ceil(array_size[2] / 2)
    z_start = np.ceil(-array_size[0] / 2)
    z_end = np.ceil(array_size[0] / 2)
    z, y, x = np.mgrid[z_start:z_end, y_start:y_end, x_start:x_end]
    return np.fft.ifftshift(x ** 2 + y ** 2 + z ** 2 <= radius ** 2)


def convolve(image_1, image_2):
    """Perform a convolution of two 2D images of equal shape via convolution theorem

    Args:
        image_1 (ndarray): An image
        image_2 (ndarray): An image

    Returns:
        (ndarray): An image

    """
    return np.fft.ifftn(np.fft.fftn(image_1) * np.fft.fftn(image_2))


def lmr(image, filter_type=None, radius=None):
    """Perform local mean removal. Default to using a circle of radius radius, otherwise use the kernel provided in
    filter. The kernel must be the same shape as image if not None.

    Args:
        image (ndarray): A 2D image
        filter_type (None or ndarray): The filter to apply to find the local mean, typically a shape centred at [0, 0]
        in an ndarray (e.g. the result of np.fft.ifftshift(A shape centred in the centre of an array))
        radius (float): The radius of a circle to use as default filter if filter is None

    Returns:
        (ndarray): A 2D image

    """
    if filter_type is None:
        if not isinstance(radius, int):
            raise ValueError(
                "You must provide a numerical radius for LMR if filter_type is None"
            )
        if len(image.shape) == 2:
            filt = circle(image.shape, radius=radius)
        elif len(image.shape) == 3:
            filt = sphere(image.shape, radius=radius)
    else:
        filt = filter_type
    filtered_image = convolve(image, filt) / np.sum(filt)
    lmr_image = image - np.abs(filtered_image)
    # remove machine precision errors
    lmr_image[np.abs(lmr_image) < 1e-10] = 0
    return lmr_image


def _final_hist_pre_peak_val(arr):
    """Find a value at which 'most' pixels in an array arr are below, where pixels above this value are rare. Extract
    this by finding the convex decreasing knee in a smoothed histogram of pixel values

    Args:
        arr (ndarray): typically an image

    Returns:
        (float): the value above which pixels are scarce

    """
    if len(arr.shape) > 1:
        arr = np.copy(arr).reshape(-1)
    hist = np.histogram(arr, bins=100)
    smoothed_hist_counts = ndimage.gaussian_filter1d(hist[0].astype("float"), sigma=3)
    knee_locator = KneeLocator(
        hist[1][1:], smoothed_hist_counts, S=1.0, curve="convex", direction="decreasing"
    )
    return knee_locator.knee


def _zero_crossings_2d(image, thresh=0):
    """Get a binary array the shape of 2D image, with True where zero crossing points occur

    Args:
        image (ndarray): A 2D image
        thresh (float or 'auto'): The amount by which adjacent pixel values must differ while crossing 0 to be counted
        as a crossing point.

    Returns:
        (ndarray): Binary array the shape of image, with True where zero crossing points occur

    """
    crossings = np.zeros([shape_element - 1 for shape_element in image.shape])
    for y_shift in range(2):
        for x_shift in range(2):
            if y_shift == 0 and x_shift == 0:
                continue
            y_end = image.shape[0] - 1 + y_shift
            x_end = image.shape[1] - 1 + x_shift
            curr_diff = np.abs(
                (image[0:-1, 0:-1] - image[y_shift:y_end, x_shift:x_end])
            )
            if thresh == "auto":
                curr_thresh = _final_hist_pre_peak_val(curr_diff)
            else:
                curr_thresh = thresh
            curr_diff = curr_diff > curr_thresh
            sign_change = (image[0:-1, 0:-1] * image[y_shift:y_end, x_shift:x_end]) < 0
            curr_crossings = curr_diff * sign_change
            crossings = np.logical_or(crossings, curr_crossings)
    crossings = np.pad(crossings, (0, 1), mode="constant")
    return crossings


def _zero_crossings_3d(image, thresh=0):
    """Get a binary array the shape of 3D image, with True where zero crossing points occur

    Args:
        image (ndarray): A 3D image
        thresh (float): The amount by which adjacent pixel values must differ while crossing 0 to be counted as a
        crossing point

    Returns:
        (ndarray): Binary array the shape of image, with True where zero crossing points occur

    """
    crossings = np.zeros([shape_element - 1 for shape_element in image.shape])
    for y_shift in range(2):
        for x_shift in range(2):
            for z_shift in range(2):
                if y_shift == 0 and x_shift == 0 and z_shift == 0:
                    continue
                y_end = image.shape[0] - 1 + y_shift
                x_end = image.shape[1] - 1 + x_shift
                z_end = image.shape[2] - 1 + z_shift
                curr_diff = np.abs(
                    (
                        image[0:-1, 0:-1, 0:-1]
                        - image[y_shift:y_end, x_shift:x_end, z_shift:z_end]
                    )
                )
                if thresh == "auto":
                    curr_thresh = _final_hist_pre_peak_val(curr_diff)
                else:
                    curr_thresh = thresh
                curr_diff = curr_diff > curr_thresh
                sign_change = (
                    image[0:-1, 0:-1, 0:-1]
                    * image[y_shift:y_end, x_shift:x_end, z_shift:z_end]
                ) < 0
                curr_crossings = curr_diff * sign_change
                crossings = np.logical_or(crossings, curr_crossings)
    crossings = np.pad(crossings, (0, 1), mode="constant")
    return crossings


def zero_crossings(image, thresh=0):
    """Get a binary array the shape of image, with True where zero crossing points occur

    Args:
        image (ndarray): A 2D image
        thresh (float or 'auto'): The amount by which adjacent pixel values must differ while crossing 0 to be counted
        as a crossing point. If 'auto', chooses the bin edge value prior to the maximum counts in the smoothed histogram
        of pixel values

    Returns:
        (ndarray): Binary array the shape of image, with True where zero crossing points occur

    """
    if len(image.shape) == 2:
        crossings = _zero_crossings_2d(image=image, thresh=thresh)
    elif len(image.shape) == 3:
        crossings = _zero_crossings_3d(image=image, thresh=thresh)
    return crossings


def pad_nd(image, shape):
    """Pad an image up to shape with its mean value, where padding is added at end of each axis

    Args:
        image (ndarray): An n-dimensional image
        shape (list of ints): expected output shape, must but greater than image.shape in each dimension

    Returns:
        (ndarray): Padded image

    """
    pad_widths = tuple((0, shape[i] - image.shape[i]) for i in range(len(shape)))
    return np.pad(image, pad_width=pad_widths, constant_values=image.mean())


def max_shape_from_image_list(images):
    """Get the maximum size in each dimension from a list of images, useful for defining an empty image that can be
    used to overlay every image

    Args:
        images (list of ndarrays): list of images, all of same dimensionality

    Returns:
        (tuple of ints): The maximum shape element along each dimension across all images

    """
    return tuple(
        max([im.shape[i] for im in images]) for i in range(len(images[0].shape))
    )


def generate_overlay_2d(images, normalize=True):
    """Take two grayscale images and overlay them in R and G channels of an output image, where output image is large
    enough to fit all data from both images

    Args:
        images (list of ndarrays): list of 2D images
        normalize (bool): whether to divide by max of each image while assigning to a layer of output array

    Returns:
        (ndarray): 3D array, [rows, cols, channels], first two channels filled by input images

    """
    overlaid_shape = max_shape_from_image_list(images)
    overlaid_shape += (3,)
    overlaid = np.zeros(overlaid_shape, dtype=images[0].dtype)
    for im_i, im in enumerate(images):
        if normalize is True and np.abs(im.max()) > 0:
            overlaid[: im.shape[0], : im.shape[1], im_i] = im / im.max()
        else:
            overlaid[: im.shape[0], : im.shape[1], im_i] = im

    return overlaid
