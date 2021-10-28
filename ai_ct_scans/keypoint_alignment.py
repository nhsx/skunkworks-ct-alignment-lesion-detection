"""Module to provide functions to perform keypoint detection alignment."""
import cv2
import numpy as np


def align_image(image, reference, detector="SIFT"):
    """Align an image to a reference image using keypoint detection.

    Args:
        image (np.ndarray): A 2D greyscale image to be aligned.
        reference (np.ndarray): A 2D greyscale image that to be used as the reference.
        detector (str): The keypoint detector to use. Should be one of ['SIFT', 'ORB'].

    Returns:
        (np.ndarray): A 2D greyscale image that is the result of the alignment.
    """
    keypoints_image, descriptors_image = get_keypoints_and_descriptors(image, detector)
    keypoints_reference, descriptors_reference = get_keypoints_and_descriptors(
        reference, detector
    )
    matched_descriptors = match_descriptors(descriptors_reference, descriptors_image)
    homography = find_homography(
        keypoints_reference, keypoints_image, matched_descriptors
    )
    aligned = cv2.warpPerspective(
        image, np.linalg.inv(homography[0]), (reference.shape[1], reference.shape[0])
    )

    return aligned


def find_homography(key_points_1, key_points_2, good_matches):
    """Estimate the transformation that maps one set of keypoints to another.

    Args:
        key_points_1 (list of cv2.KeyPoint): The keypoints extracted from the source image.
        key_points_2 (list of cv2.KeyPoint): The keypoints extracted from the destination image.
        good_matches (list of cv2.DMatch): Matched descriptors.

    Returns:
        (np.ndarray): A transformation matrix that maps the first set of keypoints to the second.
        (np.ndarray): Mask of inlier and outlier points used to estimate transformation.
    """
    src_points = np.float32(
        [key_points_1[m.queryIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)
    dst_points = np.float32(
        [key_points_2[m.trainIdx].pt for m in good_matches]
    ).reshape(-1, 1, 2)

    return cv2.findHomography(src_points, dst_points, cv2.RANSAC, 3.0)


def match_descriptors(descriptors_1, descriptors_2):
    """Match descriptors that have come from two separate images.

    Args:
        descriptors_1 (np.ndarray): Descriptors that correspond to keypoints extracted from the first image.
        descriptors_2 (np.ndarray): Descriptors that correspond to keypoints extracted from the first image.

    Returns:
        (list of cv2.DMatch): Matched descriptors.
    """
    flann = cv2.FlannBasedMatcher({"algorithm": 0, "trees": 5}, {"checks": 100})
    matches = flann.knnMatch(np.float32(descriptors_1), np.float32(descriptors_2), k=2)
    good_matches = sieve_matches_lowe(matches)

    return good_matches


def sieve_matches_lowe(matches):
    """Apply Lowe's ratio test to retain only good matches.

    Args:
        matches: (list of cv2.DMatch): Matched descriptors.

    Returns:
        (list of cv2.DMatch): Matched descriptors that passed the test.
    """
    good_matches = []
    for m, n in matches:
        if m.distance < n.distance * 0.8:
            good_matches.append(m)

    return good_matches


def get_keypoints_and_descriptors(image, detector="SIFT"):
    """Get keypoints and compute corresponding descriptors from an image.

    Args:
        image (np.ndarray): A 2D greyscale image.
        detector (str): The keypoint detector to use. Should be one of ['SIFT', 'ORB'].

    Returns:
        (list): List of keypoints as cv2.KeyPoint
        (list): Descriptor corresponding to the returned keypoints
    """
    detector = (
        cv2.SIFT_create(edgeThreshold=15, sigma=1.2)
        if detector == "SIFT"
        else cv2.ORB_create(fastThreshold=25)
    )

    key_points, descriptors = detector.detectAndCompute(image, None)

    return key_points, descriptors
