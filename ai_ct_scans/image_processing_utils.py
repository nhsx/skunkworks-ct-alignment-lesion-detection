"""Module containing generic image processing functionality."""
import cv2


def normalise(image):
    """Normalise an image.

    Args:
        image (np.ndarray): A 2D greyscale image.

    Returns:
        (np.ndarray): A 2D greyscale image with 8 bit intensity values.
    """
    return cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")


def overlay_warp_on_slice(aligned_slice, warp_slice):
    """Generate an image showing a 2D aligned scan slice with a heatmap overlaid representing the magnitude of the
    warp at each pixel.

    Args:
        aligned_slice (np.ndarray): A 2D greyscale image from a scan.
        warp_slice (np.ndarray): A 2D greyscale array representing the magnitude of the warp. Should be the same shape
            as aligned_slice.

    Returns:
        (np.ndarray): An image of the 2D scan slice with warp heatmap overlaid.
    """
    grey = cv2.cvtColor(normalise(aligned_slice), cv2.COLOR_GRAY2RGB)
    overlay = cv2.applyColorMap(normalise(warp_slice), cv2.COLORMAP_JET)
    overlay = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
    grey_with_overlay = cv2.addWeighted(grey, 0.5, overlay, 0.5, 0)
    return grey_with_overlay
