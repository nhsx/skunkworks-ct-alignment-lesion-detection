"""Module to provide functions to perform non-rigid alignment of images."""
import argparse
import math
import pickle
import cv2

import cycpd
import numpy as np
from scipy import ndimage
from skimage import transform
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from ai_ct_scans import data_loading
from ai_ct_scans import phase_correlation_image_processing
from ai_ct_scans.point_matching import match_indices


def align_3D_using_CPD(
    image,
    reference,
    match_filter_distance=1.0,
    maximum_source_points=5000,
    maximum_target_points=30000,
):
    """Align a full 3d scan to a reference using a non-rigid transform.

    Args:
        image (np.ndarray): A 3D scan to be aligned.
        reference (np.ndarray): A 3D scan to be used as the reference.
        match_filter_distance (float): Distance threshold used to filter points following matching after coherent
            point drift.
        maximum_source_points (int): Maximum number of points to use from scan during alignment.
        maximum_target_points (int): Maximum number of points to use from reference scan during alignment.

    Returns:
        (np.ndarray): A 3D greyscale image that is the result of the alignment.
    """
    trans = estimate_3D_alignment_transform(
        image,
        reference,
        match_filter_distance=match_filter_distance,
        maximum_source_points=maximum_source_points,
        maximum_target_points=maximum_target_points,
    )
    return transform_3d_volume(image, trans.predict)


def estimate_3D_alignment_transform(
    image,
    reference,
    match_filter_distance=1.0,
    maximum_source_points=5000,
    maximum_target_points=30000,
):
    """Estimate the transform required to align a full 3d scan to a reference using a non-rigid transform.

    Args:
        image (np.ndarray): A 3D scan to be aligned.
        reference (np.ndarray): A 3D scan to be used as the reference.
        match_filter_distance (float): Distance threshold used to filter points.
        maximum_source_points (int): Maximum number of points to use from scan during alignment.
        maximum_target_points (int): Maximum number of points to use from reference scan during alignment.

    Returns:
        (np.ndarray): A 3D greyscale image that is the result of the alignment.
    """
    _remove_table(reference)
    _remove_table(image)
    target_points = _extract_points_3d(reference)
    source_points = _extract_points_3d(image)

    # Filter points if required
    if target_points.shape[0] > maximum_target_points:
        target_points = target_points[
            :: target_points.shape[0] // maximum_target_points
        ]
    if source_points.shape[0] > maximum_source_points:
        source_points = source_points[
            :: source_points.shape[0] // maximum_source_points
        ]

    aligned_points = _align_points(target_points, source_points)
    matched_target_points, matched_source_points = _filter_matches(
        target_points, source_points, aligned_points, match_filter_distance
    )
    return _calculate_transform_between_points(
        matched_target_points, matched_source_points
    )


def transform_3d_volume_in_chunks(image, transform_function, chunk_thickness):
    """Transform a 3d volume by a generic geometric transform, chunk-by-chunk in order to reduce required memory.

    Chunks are taken in the axial direction.

    Args:
        image (np.ndarray): A 3D greyscale image to transform.
        transform_function (callable): Function used to map coordinates in input image to transformed image. Takes
        and returns points as an np.ndarray of shape (n_samples, 3).
        chunk_thickness (int): Thickness of the chunks to align

    Returns:
        (np.ndarray): A 3D greyscale image that is the result of the transform.
    """
    transformed = np.empty_like(image)
    _, y_len, z_len = image.shape
    for i in range(math.ceil(image.shape[0] / chunk_thickness)):
        chunk_start = i * chunk_thickness
        chunk_end = min((i + 1) * chunk_thickness, image.shape[0])
        x_len = chunk_end - chunk_start
        grid_points = _get_grid_points((x_len, y_len, z_len), offset=chunk_start)
        coords_in_input = transform_function(grid_points)
        coords_in_input = np.array(
            [
                coords_in_input[:, 0].reshape(x_len, y_len, z_len),
                coords_in_input[:, 1].reshape(x_len, y_len, z_len),
                coords_in_input[:, 2].reshape(x_len, y_len, z_len),
            ]
        )
        transformed[chunk_start:chunk_end, :, :] = ndimage.map_coordinates(
            image, coords_in_input, order=0
        )

    return transformed


def transform_3d_volume(image, transform_function):
    """Transform a 3d volume by a generic geometric transform.

    Args:
        image (np.ndarray): A 3D greyscale image to transform.
        transform_function (callable): Function used to map coordinates in input image to transformed image. Takes
        and returns points as an np.ndarray of shape (n_samples, 3).

    Returns:
        (np.ndarray): A 3D greyscale image that is the result of the transform.
    """
    x_len, y_len, z_len = image.shape
    grid_points = _get_grid_points((x_len, y_len, z_len))
    coords_in_input = transform_function(grid_points)
    coords_in_input = np.array(
        [
            coords_in_input[:, 0].reshape(x_len, y_len, z_len),
            coords_in_input[:, 1].reshape(x_len, y_len, z_len),
            coords_in_input[:, 2].reshape(x_len, y_len, z_len),
        ]
    )
    aligned = ndimage.map_coordinates(image, coords_in_input, order=0)
    return aligned


def write_transform(transform, path):
    """Serialise a transform and write to disk.

    Args:
        transform (inst of sklearn.pipeline.Pipeline): A transform that maps points in the source image onto the
            reference image.
        path (str): Path to write serialised transform to.
    """
    if not str(path).endswith(".pkl"):
        raise ValueError("Extension must be .pkl")

    with open(path, "wb") as f:
        pickle.dump(transform, f)


def read_transform(path):
    """Read and load a serialised transform from disk.

    Args:
        path (str or Path): Path to write serialised transform to.

    Returns:
        (inst of sklearn.pipeline.Pipeline): A transform that maps points in the source image onto the reference image.
    """
    if not str(path).endswith(".pkl"):
        raise ValueError("Extension must be .pkl")

    with open(path, "rb") as f:
        trans = pickle.load(f)

    return trans


def align_2D_using_CPD(image, reference, point_threshold=None, filter_distance=1.0):
    """Align an 2D image to a reference image using a non-rigid transform.

    Args:
        image (np.ndarray): A 2D greyscale image to be aligned.
        reference (np.ndarray): A 2D greyscale image that to be used as the reference.
        point_threshold (np.ndarray): Threshold value used for point extraction as an absolute intensity value.
        filter_distance (float): Distance threshold used to filter matched points.

    Returns:
        (np.ndarray): A 2D greyscale image that is the result of the alignment.
    """
    target_points = _extract_points_2d(reference, point_threshold)
    source_points = _extract_points_2d(image, point_threshold)
    aligned_points = _align_points(target_points, source_points)
    matched_target_points, matched_source_points = _filter_matches(
        target_points, source_points, aligned_points, filter_distance
    )
    trans = _calculate_transform_between_points(
        matched_target_points, matched_source_points
    )
    non_rigid_aligned = transform.warp(image, trans.predict)
    return non_rigid_aligned


def get_warp_overlay(shape, transform_function, chunk_thickness=10):
    """Generate a heatmap showing the the magnitude of the warp used to align a scan at each voxel.

    The warp is defined as the fractional volumetric change at each voxel.

    Args:
        shape (tuple of int): The shape of the scan that has been aligned
        transform_function (callable): The transform function used to align the scan.
        chunk_thickness (int): Thickness of the chunks used to build up the warp overlay.

    Returns:
        (np.ndarray): 3D heatmap representing the warp magnitude.
    """
    warp_overlay = np.zeros(shape, dtype=np.float32)
    _, y_len, z_len = shape
    for i in range(math.ceil(shape[0] / chunk_thickness)):
        chunk_start = i * chunk_thickness
        chunk_end = min((i + 1) * chunk_thickness, shape[0])
        x_len = chunk_end - chunk_start
        grid_points = _get_grid_points((x_len, y_len, z_len), offset=chunk_start)
        coords_in_input = transform_function(grid_points)
        shift = coords_in_input - grid_points
        shift = shift.reshape(x_len, y_len, z_len, 3)
        e11 = np.gradient(shift[:, :, :, 0], axis=0)
        e22 = np.gradient(shift[:, :, :, 1], axis=1)
        e33 = np.gradient(shift[:, :, :, 2], axis=2)
        warp_chunk = (
            e11 + e22 + e33 + e11 * e22 + e11 * e33 + e22 * e33 + e11 * e22 * e33
        )
        warp_chunk = np.reciprocal(warp_chunk + 1) - 1
        warp_overlay[chunk_start:chunk_end, :, :] = warp_chunk

    return warp_overlay


def _extract_points_2d(image, threshold=None):
    """Extract a set of keypoints from an image.

    Args:
        image (np.ndarray): A 2D greyscale image to extract points from.
        threshold (float): Threshold to take points above.

    Returns:
        (np.ndarray): A set of points of shape (n, 2).
    """
    image_filtered = phase_correlation_image_processing.lmr(
        image, filter_type=None, radius=3
    )
    if not threshold:
        threshold = np.percentile(image_filtered, 99)
    key_point_coords = np.where(image_filtered[2:-2, 2:-2] > threshold)
    points = np.stack(key_point_coords, axis=1).astype(np.float64)
    points = points[:, [1, 0]]
    return points


def _extract_points_3d(image, threshold=None):
    """Extract a set of keypoints from an image.

    Args:
        image (np.ndarray): A 2D greyscale image to extract points from.
        threshold (float): Threshold to take points above.

    Returns:
        (np.ndarray): A set of points of shape (n, 3).
    """
    image_filtered = phase_correlation_image_processing.lmr(
        image, filter_type=None, radius=10
    )
    if not threshold:
        threshold = np.percentile(image_filtered, 98)
    key_point_coords = np.where(image_filtered[2:-2, 2:-2, 2:-2] > threshold)
    points = np.stack(key_point_coords, axis=1).astype(np.float64)
    return points


def _align_points(target_points, source_points):
    """Align a set of points to a set of target points.

    Args:
        target_points (np.ndarray): A set of points that define the target, of shape (n_samples, n_dimensions).
        source_points (np.ndarray): A set of points to align, of shape (n_samples, n_dimensions).

    Returns:
        (np.ndarray): A set of aligned points of shape (n, 2).
    """
    reg = cycpd.deformable_registration(
        **{
            "X": target_points,
            "Y": source_points,
            "alpha": 0.05,
            "beta": 30,
            "max_iterations": 1000,
        }
    )
    non_rigid_out = reg.register()
    return non_rigid_out[0]


def _filter_matches(target_points, source_points, aligned_points, thresh):
    """Match and filter points from a target and source set based on aligned points.

    Args:
        target_points (np.ndarray): A set of points that define the target, of shape (n_samples, n_dimensions).
        source_points (np.ndarray): A set of points to align, of shape (n_samples, n_dimensions).
        aligned_points (np.ndarray): The source points after alignment to the target points, of shape
            (n_samples, n_dimensions).
        thresh (float): Distance threshold used to filter points.

    Returns:
        (np.ndarray): A subset of the target_points which have been matched and filtered, of shape
            (n_samples, n_dimensions).
        (np.ndarray): A subset of the source_points which have been matched and filtered, of shape
            (n_samples, n_dimensions).
    """
    matched_indices = match_indices(target_points, aligned_points)
    target_points = target_points[matched_indices[1]]
    source_points = source_points[matched_indices[0]]
    dists = np.linalg.norm(target_points - aligned_points[matched_indices[0]], axis=1)
    target_points_filtered = target_points[np.where(dists < thresh)]
    source_points_filtered = source_points[np.where(dists < thresh)]
    return target_points_filtered, source_points_filtered


def _calculate_transform_between_points(target_points, source_points):
    """Calculate a transform that maps matched sets of points.

    Args:
        target_points (np.ndarray): A set of matched points from the target image, of shape (n_samples, n_dimensions).
        source_points (np.ndarray): A set of matched points from the source image, of shape (n_samples, n_dimensions).

    Returns:
        (inst of sklearn.pipeline.Pipeline): A transform that maps points in the source image onto the reference image.
    """
    poly_trans = make_pipeline(PolynomialFeatures(degree=3), Ridge())
    poly_trans.fit(target_points, source_points)
    return poly_trans


def _get_grid_points(shape, offset=0):
    """Get an array of coordinates for each point in 3D scan.

    Args:
        shape (tuple): The shape of the 3D scan.
        offset (int): A constant offset to be applied to the first dimension of each coordinate.

    Returns:
        (np.ndarray): Array of grid points of shape (n, 3).
    """
    x_grid, y_grid, z_grid = np.meshgrid(
        np.arange(shape[0], dtype=np.uint16) + offset,
        np.arange(shape[1], dtype=np.uint16),
        np.arange(shape[2], dtype=np.uint16),
        indexing="ij",
        copy=False,
    )
    grid_points = np.stack([x_grid.ravel(), y_grid.ravel(), z_grid.ravel()], axis=1)
    return grid_points


def _remove_table(scan, margin=20):
    """Detect the table in a scan and set everything below the top to zero.

    Args:
        scan (np.ndarray): A 3D scan.
    """
    sagittal_mid = scan.shape[2] // 2
    for i in range(sagittal_mid, scan.shape[2]):
        table_location = _find_table_in_slice(scan[:, :, i])
        if table_location:
            scan[:, table_location - margin :, i] = 0
        else:
            break

    for i in range(sagittal_mid - 1, 0, -1):
        table_location = _find_table_in_slice(scan[:, :, i])
        if table_location:
            scan[:, table_location - margin :, i] = 0
        else:
            break


def _find_table_in_slice(slice_image):
    """Find the top of the table in a single sagittal slice from a scan.

    Args:
        slice_image (np.ndarray): A 2D slice from a scan in the sagittal plane.

    Returns:
        (int): Coordinate in the coronal axis of the highest detected table part, or None if no table is detected.
    """
    _, binary_slice = cv2.threshold(slice_image, 800, 1, cv2.THRESH_BINARY)
    edges = cv2.Sobel(binary_slice, cv2.CV_8U, 1, 0, ksize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, int(edges.shape[0] * 0.5), None, 0, 0)
    if lines is None or lines.size == 0:
        return
    return int(min(lines[:, 0, 0]))


def _divergence(f):
    """Calculate the divergence of a vector field.

    Args:
        f (list of np.ndarray): Components of the vector field.

    Returns:
        (np.ndarray): The calculated divergence, as an array with the same shape as each element in f.
    """
    num_dims = len(f)
    return np.ufunc.reduce(np.add, [np.gradient(f[i], axis=i) for i in range(num_dims)])


def main():
    """CLI tool to calculate or apply a transform to align 3D scans."""
    parser = argparse.ArgumentParser(
        description="Perform non-rigid alignment on a 3D scan and write transform to "
        "disk."
    )
    parser.add_argument("patient", type=int, help="Number of the patient")
    parser.add_argument(
        "output_path", type=str, help="Path to write transform to.  Must end in .pkl"
    )
    parser.add_argument(
        "--source_points",
        type=int,
        help="Maximum number of points to extract from scan during " "alignment",
        default=5000,
    )
    parser.add_argument(
        "--target_points",
        type=int,
        help="Maximum number of points to extract from reference scan during alignment",
        default=30000,
    )
    parser.add_argument(
        "--match_filter_distance",
        type=float,
        help="Distance threshold used to filter points following matching after coherent point drift",
        default=1.0,
    )
    args = parser.parse_args()

    patient_dir = data_loading.data_root_directory() / str(args.patient)
    patient_loader = data_loading.PatientLoader(patient_dir)
    patient_loader.abdo.scan_1.load_scan()
    patient_loader.abdo.scan_2.load_scan()
    trans = estimate_3D_alignment_transform(
        patient_loader.abdo.scan_2.full_scan,
        patient_loader.abdo.scan_1.full_scan,
        match_filter_distance=args.match_filter_distance,
        maximum_source_points=args.source_points,
        maximum_target_points=args.target_points,
    )
    write_transform(trans, args.output_path)


if __name__ == "__main__":
    main()
