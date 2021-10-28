import copy
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from ai_ct_scans import data_loading
from ai_ct_scans import image_processing_utils
from ai_ct_scans import keypoint_alignment


@pytest.fixture()
def abdo_loader():
    return data_loading.BodyPartLoader(
        root_dir=Path(__file__).parent / "fixtures" / "dicom_data" / "1",
        body_part="Abdo",
    )


@pytest.fixture()
def abdo_image(abdo_loader):
    abdo_loader.scan_1.load_scan()
    return image_processing_utils.normalise(abdo_loader.scan_1.full_scan[0])


@pytest.fixture()
def simple_square_image():
    test_image = np.zeros((1000, 1000), dtype=np.uint8)
    test_image[450:550, 450:550] = 255
    return test_image


@pytest.fixture()
def simple_rectangle_image():
    test_image = np.zeros((800, 600), dtype=np.uint8)
    test_image[450:500, 450:500] = 50
    test_image[500:550, 450:500] = 100
    test_image[450:500, 500:550] = 150
    test_image[500:550, 500:550] = 200
    return test_image


@pytest.fixture()
def simple_square_image_2():
    test_image = np.zeros((1000, 1000), dtype=np.uint8)
    test_image[450:500, 450:500] = 50
    test_image[500:550, 450:500] = 100
    test_image[450:500, 500:550] = 150
    test_image[500:550, 500:550] = 200
    return test_image


class TestKeypointDetection:
    def test_keypoints_are_returned(self, abdo_image):
        output = keypoint_alignment.get_keypoints_and_descriptors(abdo_image)
        assert output[0]
        for point in output[0]:
            assert type(point) is cv2.KeyPoint

    def test_corresponding_descriptors_are_returned(self, abdo_image):
        output = keypoint_alignment.get_keypoints_and_descriptors(abdo_image)
        assert len(output[0]) == len(output[1])

    def test_keypoints_around_center_feature(self, simple_square_image):
        output = keypoint_alignment.get_keypoints_and_descriptors(simple_square_image)

        # Check that keypoints have been detected
        assert output

        # Check that the keypoints are close to the center of the image
        for point in output[0]:
            assert math.hypot(point.pt[0] - 500, point.pt[1] - 500) < 100


class TestKeypointAlignment:
    def test_returned_image_is_same_size_as_input(self, simple_rectangle_image):
        aligned_image = keypoint_alignment.align_image(
            simple_rectangle_image, simple_rectangle_image
        )
        assert aligned_image.shape == simple_rectangle_image.shape

    def test_simple_image_is_aligned(self, simple_square_image_2):
        reference_image = simple_square_image_2

        # generate an image subtlety different to the reference
        image_pre_shift = copy.deepcopy(reference_image)
        image_pre_shift[400, 400] = 111

        shifted_image = np.roll(image_pre_shift, 200, axis=0)
        shifted_image = np.roll(shifted_image, 200, axis=1)

        np.testing.assert_array_equal(
            keypoint_alignment.align_image(shifted_image, reference_image),
            image_pre_shift,
        )

    def test_orb_can_be_used_for_keypoint_detection(self, simple_square_image_2):
        reference_image = simple_square_image_2

        # generate an image subtlety different to the reference
        image_pre_shift = copy.deepcopy(reference_image)
        image_pre_shift[400, 400] = 111

        shifted_image = np.roll(image_pre_shift, 200, axis=0)
        shifted_image = np.roll(shifted_image, 200, axis=1)

        # This example doesn't align perfectly using ORB, so this tests that it runs correctly and aligns approximately
        aligned = keypoint_alignment.align_image(shifted_image, reference_image, "ORB")
        assert np.count_nonzero(aligned != image_pre_shift) < 1000
