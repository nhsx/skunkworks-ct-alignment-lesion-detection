import os
import pickle

import mock
import numpy as np
import pytest
from scipy import ndimage
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import PolynomialFeatures

from ai_ct_scans import non_rigid_alignment


@pytest.fixture()
def target_points():
    return np.random.rand(10, 3)


@pytest.fixture()
def source_points():
    return np.random.rand(10, 3)


@pytest.fixture()
def simple_scan():
    scan = np.zeros((100, 100, 100), dtype=np.float64)
    scan[45:55, 45:55, 45:55] = 1000
    return scan


class TestNonRigidAlignment2D:
    @pytest.fixture()
    def simple_rectangle_image(self):
        test_image = np.zeros((1000, 1000), dtype=np.float64)
        test_image[450:500, 450:500] = 50
        test_image[500:550, 450:500] = 100
        test_image[450:500, 500:550] = 150
        test_image[500:550, 500:550] = 200
        return test_image

    def test_returned_image_is_same_dimensions(self, simple_rectangle_image):
        shifted = np.roll(simple_rectangle_image, 10, axis=0)
        aligned = non_rigid_alignment.align_2D_using_CPD(
            shifted, simple_rectangle_image, point_threshold=50, filter_distance=10.0
        )
        assert shifted.shape == aligned.shape

    def test_alignment_of_shifted_image(self, simple_rectangle_image):
        shifted = np.roll(simple_rectangle_image, 20, axis=0)
        aligned = non_rigid_alignment.align_2D_using_CPD(
            shifted, simple_rectangle_image, point_threshold=50, filter_distance=10.0
        )
        # Check that the alignment has improved
        assert np.count_nonzero(aligned != simple_rectangle_image) < np.count_nonzero(
            shifted != simple_rectangle_image
        )

    def test_on_simple_warped_image(self, simple_rectangle_image):
        def apply_poly(xy):
            return xy[0], xy[1] - 0.1 * xy[0] + 0.0001 * xy[0] ** 2

        warped = ndimage.geometric_transform(simple_rectangle_image, apply_poly)

        aligned = non_rigid_alignment.align_2D_using_CPD(
            warped, simple_rectangle_image, point_threshold=50, filter_distance=10.0
        )
        # Check that the alignment has improved
        assert np.count_nonzero(aligned != simple_rectangle_image) < np.count_nonzero(
            warped != simple_rectangle_image
        )


class TestNonRigidAlignment3D:
    def test_returned_scan_is_same_size(self, simple_scan):
        shifted_scan = np.roll(simple_scan, (5, 10, 20), (0, 1, 2))
        aligned = non_rigid_alignment.align_3D_using_CPD(shifted_scan, simple_scan)
        assert aligned.shape == shifted_scan.shape

    def test_alignment_of_shifted_scan(self, simple_scan):
        shifted = np.roll(simple_scan, 20, axis=0)
        aligned = non_rigid_alignment.align_3D_using_CPD(
            shifted, simple_scan, match_filter_distance=10.0
        )
        np.testing.assert_array_equal(aligned, simple_scan)

    @pytest.mark.parametrize("maximum", (1001, 1000, 500, 200, 10))
    def test_target_points_filtered_when_above_maximum(self, simple_scan, maximum):
        shifted = np.roll(simple_scan, 20, axis=0)
        with mock.patch.object(
            non_rigid_alignment, "_align_points"
        ) as mock_align_points:
            mock_align_points.side_effect = lambda target, source: source
            non_rigid_alignment.align_3D_using_CPD(
                shifted,
                simple_scan,
                match_filter_distance=25.0,
                maximum_target_points=maximum,
            )
            assert mock_align_points.call_args_list[0][0][0].shape[0] <= maximum

    @pytest.mark.parametrize("maximum", (1001, 1000, 500, 200, 10))
    def test_source_points_filtered_when_above_maximum(self, simple_scan, maximum):
        shifted = np.roll(simple_scan, 20, axis=0)
        with mock.patch.object(
            non_rigid_alignment, "_align_points"
        ) as mock_align_points:
            mock_align_points.side_effect = lambda target, source: source
            non_rigid_alignment.align_3D_using_CPD(
                shifted,
                simple_scan,
                match_filter_distance=25.0,
                maximum_source_points=maximum,
            )
            assert mock_align_points.call_args_list[0][0][1].shape[0] <= maximum


class TestTransform3dVolume:
    def test_returned_image_is_same_size_as_input(self):
        image = np.random.rand(5, 6, 7)

        def transform(input_coords):
            return input_coords

        transformed = non_rigid_alignment.transform_3d_volume(image, transform)
        assert transformed.shape == image.shape

    def test_simple_shift(self, simple_scan):
        shifted_image = np.roll(simple_scan, (5, 10, 20), (0, 1, 2))

        def transform(input_coords):
            return input_coords + [5, 10, 20]

        transformed = non_rigid_alignment.transform_3d_volume(shifted_image, transform)
        np.testing.assert_array_equal(transformed, simple_scan)

    @pytest.mark.parametrize("chunk_thickness", (5, 10, 40, 50, 200))
    def test_transform_in_chunks(self, simple_scan, chunk_thickness):
        shifted_image = np.roll(simple_scan, (5, 10, 20), (0, 1, 2))

        def transform(input_coords):
            return input_coords + [5, 10, 20]

        transformed = non_rigid_alignment.transform_3d_volume_in_chunks(
            shifted_image, transform, chunk_thickness
        )
        np.testing.assert_array_equal(transformed, simple_scan)


class TestWriteTransform:
    def test_transform_written_to_specified_location(
        self, target_points, source_points, tmpdir
    ):
        trans = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        trans.fit(target_points, source_points)
        path = tmpdir.join("trans.pkl")
        non_rigid_alignment.write_transform(trans, path)
        assert os.path.isfile(path)

    def test_correct_parameters_written_to_file(
        self, target_points, source_points, tmpdir
    ):
        new_points = np.random.rand(2, 3)
        trans = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        trans.fit(target_points, source_points)
        transformed_points = trans.predict(new_points)
        path = tmpdir.join("trans.pkl")
        non_rigid_alignment.write_transform(trans, path)

        with open(path, "rb") as f:
            loaded_transform = pickle.load(f)

        np.testing.assert_array_equal(
            loaded_transform.predict(new_points), transformed_points
        )

    def test_exception_raised_if_path_has_incorrect_extension(self, tmpdir):
        trans = make_pipeline(PolynomialFeatures(degree=1), LinearRegression())
        path = tmpdir.join("trans")
        with pytest.raises(ValueError):
            non_rigid_alignment.write_transform(trans, path)


class TestReadTransform:
    def test_transform_read_from_location(self, target_points, source_points, tmpdir):
        trans = make_pipeline(PolynomialFeatures(degree=3), LinearRegression())
        trans.fit(target_points, source_points)
        path = tmpdir.join("trans.pkl")
        with open(path, "wb") as f:
            pickle.dump(trans, f)

        loaded_trans = non_rigid_alignment.read_transform(path)
        assert isinstance(loaded_trans, Pipeline)

    def test_alignment_with_loaded_transform(self, simple_scan, tmpdir):
        shifted = np.roll(simple_scan, 20, axis=0)
        transform = non_rigid_alignment.estimate_3D_alignment_transform(
            shifted, simple_scan, match_filter_distance=10.0
        )
        path = tmpdir.join("trans.pkl")
        non_rigid_alignment.write_transform(transform, path)
        loaded_transform = non_rigid_alignment.read_transform(path)
        aligned = non_rigid_alignment.transform_3d_volume(
            shifted, loaded_transform.predict
        )
        np.testing.assert_array_equal(aligned, simple_scan)

    def test_exception_raised_if_path_has_incorrect_extension(self, tmpdir):
        path = tmpdir.join("trans")
        with pytest.raises(ValueError):
            non_rigid_alignment.read_transform(path)


class TestGetWarpOverlay:
    @pytest.fixture()
    def no_change_transform(self):
        def transform(input_coords):
            return input_coords.astype(np.float64)

        return transform

    @pytest.fixture()
    def shift_transform(self):
        def transform(input_coords):
            return input_coords + [5.0, 10.0, 20.0]

        return transform

    @pytest.fixture()
    def squash_in_one_axis_transform(self):
        def transform(input_coords):
            return input_coords * [2.0, 1.0, 1.0]

        return transform

    @pytest.fixture()
    def squash_in_two_axis_transform(self):
        def transform(input_coords):
            return input_coords * [2.0, 2.0, 1.0]

        return transform

    @pytest.fixture()
    def squash_in_three_axis_transform(self):
        def transform(input_coords):
            return input_coords * [2.0, 2.0, 2.0]

        return transform

    @pytest.fixture()
    def stretch_in_two_axis_transform(self):
        def transform(input_coords):
            return input_coords * [1.0, 0.5, 0.5]

        return transform

    @pytest.fixture()
    def stretch_and_squash_transform(self):
        def transform(input_coords):
            return input_coords * [2.0, 1.0, 0.5]

        return transform

    def test_returned_structure_correct_shape(self, no_change_transform):
        shape = (50, 20, 25)
        warp = non_rigid_alignment.get_warp_overlay(shape, no_change_transform)
        assert warp.shape == shape

    def test_warp_for_transform_that_doesnt_alter_image_is_all_zero(
        self, no_change_transform
    ):
        shape = (50, 20, 25)
        warp = non_rigid_alignment.get_warp_overlay(shape, no_change_transform)
        np.testing.assert_array_equal(warp, np.zeros(shape))

    def test_warp_for_constant_shift_transform_is_all_zero(self, shift_transform):
        shape = (50, 20, 25)
        warp = non_rigid_alignment.get_warp_overlay(shape, shift_transform)
        np.testing.assert_array_equal(warp, np.zeros(shape))

    def test_warp_for_squash_one_axis_transform(self, squash_in_one_axis_transform):
        shape = (10, 10, 10)
        warp = non_rigid_alignment.get_warp_overlay(shape, squash_in_one_axis_transform)
        np.testing.assert_array_equal(warp, np.ones(shape) * -0.5)

    def test_warp_for_squash_two_axis_transform(self, squash_in_two_axis_transform):
        shape = (10, 10, 10)
        warp = non_rigid_alignment.get_warp_overlay(shape, squash_in_two_axis_transform)
        np.testing.assert_array_equal(warp, np.ones(shape) * -0.75)

    def test_warp_for_stretch_two_axis_transform(self, stretch_in_two_axis_transform):
        shape = (10, 10, 10)
        warp = non_rigid_alignment.get_warp_overlay(
            shape, stretch_in_two_axis_transform
        )
        np.testing.assert_array_equal(warp, np.ones(shape) * 3)

    def test_warp_for_stretch_and_squash_transform(self, stretch_and_squash_transform):
        shape = (10, 10, 10)
        warp = non_rigid_alignment.get_warp_overlay(shape, stretch_and_squash_transform)
        np.testing.assert_array_equal(warp, np.zeros(shape))

    def test_warp_for_squash_three_axis_transform(self, squash_in_three_axis_transform):
        shape = (10, 10, 10)
        warp = non_rigid_alignment.get_warp_overlay(
            shape, squash_in_three_axis_transform
        )
        np.testing.assert_array_equal(warp, np.ones(shape) * -0.875)
