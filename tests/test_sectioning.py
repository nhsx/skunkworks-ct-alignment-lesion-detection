import pytest
from ai_ct_scans import sectioning
import numpy as np
import mock
from pathlib import Path
import cv2
from ai_ct_scans import phase_correlation_image_processing


class TestTextonSectioner:
    @pytest.fixture()
    def sectioner(self, patched_root_directory):
        return sectioning.TextonSectioner(total_samples=100, samples_per_image=19)

    def test_sectioner_has_ndarray_filters(self, sectioner):
        for filter in sectioner.filters:
            assert isinstance(filter, np.ndarray)

    @pytest.fixture()
    def sectioner_gabor_kernel_patch(self, monkeypatch, patched_root_directory):
        monkeypatch.setattr(sectioning, "gabor_kernel", mock.MagicMock())
        return sectioning.TextonSectioner(filter_type=["gabor"])

    def test_gabor_used_if_gabor_set(self, sectioner_gabor_kernel_patch):
        assert sectioning.gabor_kernel.call_count > 0

    def test_single_image_texton_descriptors_returns_ndarray_of_correct_shape(
        self, sectioner
    ):
        im = sectioner.dataset.stream_next()
        descriptors = sectioner.single_image_texton_descriptors(im)
        assert descriptors.shape == (sectioner.total_filters, *im.shape)

    @pytest.fixture()
    def sectioner_w_texton_sample_set(self, sectioner):
        np.random.seed(555)
        sectioner.build_sample_texton_set(threshold=100, random=True)
        return sectioner

    def test_build_sample_texton_set_gets_correct_shape(
        self, sectioner_w_texton_sample_set
    ):
        assert sectioner_w_texton_sample_set.texton_sample_set.shape == (
            sectioner_w_texton_sample_set.total_samples,
            sectioner_w_texton_sample_set.total_filters,
        )

    @pytest.fixture()
    def sectioner_w_trained_clusterer(self, sectioner_w_texton_sample_set):
        sectioner_w_texton_sample_set.train_clusterers()
        return sectioner_w_texton_sample_set

    def test_clusterer_can_predict_on_new_ims(self, sectioner_w_trained_clusterer):
        np.random.seed(555)
        im = sectioner_w_trained_clusterer.dataset.stream_next(
            threshold=308, random=True
        )
        labeled_im = sectioner_w_trained_clusterer.label_im(im)
        assert im.shape == labeled_im.shape

    def test_clusterers_can_be_saved_and_reloaded(
        self, sectioner_w_trained_clusterer, tmpdir, patched_root_directory
    ):
        model_path = Path(tmpdir) / "test.pkl"
        new_sectioner = sectioning.TextonSectioner()
        np.random.seed(587)
        input_im = sectioner_w_trained_clusterer.dataset.stream_next()
        expected_im = sectioner_w_trained_clusterer.label_im(input_im)
        sectioner_w_trained_clusterer.save(model_path)
        new_sectioner.load(model_path)
        labeled_im = new_sectioner.label_im(input_im)
        np.testing.assert_array_equal(expected_im, labeled_im)


class TestMeanShiftWithProbs:
    @pytest.fixture()
    def trained_clusterer_and_train_set(self):
        np.random.seed(555)
        train_set = np.random.rand(100, 1)
        train_set[50:] += 2
        clusterer = sectioning.MeanShiftWithProbs()
        clusterer.fit(train_set)
        return clusterer, train_set

    def test_probs_predictable(self, trained_clusterer_and_train_set):
        clusterer, train_set = trained_clusterer_and_train_set
        probs = clusterer.predict_proba(train_set)
        assert isinstance(probs, np.ndarray)
        # check probability of membership to first cluster is higher for first half, higher for second cluster for
        # second half
        assert (probs[:50, 0] > probs[:50, 1]).all()
        assert (probs[50:, 0] < probs[50:, 1]).all()

    def test_probs_recoverable_for_single_cluser_label(
        self, trained_clusterer_and_train_set
    ):
        clusterer, train_set = trained_clusterer_and_train_set
        full_probs = clusterer.predict_proba(train_set)
        probs = [
            clusterer.predict_proba(train_set, cluster_label=label)
            for label in range(2)
        ]
        np.testing.assert_array_equal(probs[0], full_probs[:, 0])
        np.testing.assert_array_equal(probs[1], full_probs[:, 1])


class TestEllipseFitter:
    @pytest.fixture()
    def one_easy_ellipse(self):
        image = np.zeros([100, 150], dtype="uint8")
        centre_coords = (75, 50)
        axes_lengths = (30, 40)
        angle = 45
        image = cv2.ellipse(
            image, (centre_coords, axes_lengths, angle), color=1, thickness=-1
        )
        return image.astype("int32"), (centre_coords, axes_lengths, angle)

    @pytest.fixture()
    def one_easy_ellipse_image(self, one_easy_ellipse):
        return one_easy_ellipse[0]

    @pytest.fixture()
    def fitter(self):
        return sectioning.EllipseFitter()

    def test_easy_ellipse_found(self, one_easy_ellipse, fitter):
        ellipses, _ = fitter.fit_ellipses(one_easy_ellipse[0])
        for param_found, param_expected in zip(ellipses[0], one_easy_ellipse[1]):
            np.testing.assert_allclose(param_expected, param_found, rtol=1e-1)

    @pytest.fixture()
    def two_overlapping_ellipses(self, one_easy_ellipse):
        first_ellipse, (centre_coords, axes_lengths, angle) = one_easy_ellipse
        new_axes_lengths = (axes_lengths[0] - 10, axes_lengths[1] - 10)
        two_ellipses = cv2.ellipse(
            first_ellipse,
            (centre_coords, new_axes_lengths, angle),
            color=2,
            thickness=-1,
        )
        return (
            two_ellipses,
            (centre_coords, axes_lengths, angle),
            (centre_coords, new_axes_lengths, angle),
        )

    def test_overlapping_ellipses_found(self, two_overlapping_ellipses, fitter):
        (
            two_ellipses_image,
            ellipse_params_1,
            ellipse_params_2,
        ) = two_overlapping_ellipses
        ellipses, _ = fitter.fit_ellipses(two_ellipses_image)
        match_count = 0
        for param_sets in [ellipse_params_1, ellipse_params_2]:
            for ellipse in ellipses:
                curr_matched_params = 0
                for found_param, expected_param in zip(ellipse, param_sets):
                    if np.allclose(found_param, expected_param, rtol=1e-1):
                        curr_matched_params += 1
                    if curr_matched_params == 3:
                        match_count += 1
        # edge detection can lead to 'bulky' edges that cause multiple contours for what, by eye, you'd expect to be one
        # contour, leading to multiple ellipses found for one edge sometimes. But this is ok as long as they're genuine
        # ellipses, so match_count can be >2 when looking for 2 ellipses
        assert match_count >= 2

    def test_fitter_accepts_sequential_images_of_different_shape(
        self, one_easy_ellipse, fitter
    ):
        first_ellipse_image, _ = one_easy_ellipse
        second_ellipse_image = first_ellipse_image[:-1, :-1]
        first_ellipses, _ = fitter.fit_ellipses(first_ellipse_image)
        second_ellipses, _ = fitter.fit_ellipses(second_ellipse_image)
        for first_ellipse, second_ellipse in zip(first_ellipses, second_ellipses):
            for param_1, param_2 in zip(first_ellipse, second_ellipse):
                np.testing.assert_array_equal(param_1, param_2)

    def test_fitter_rejects_convex_ellipses(self, one_easy_ellipse, fitter):
        ellipse_image, _ = one_easy_ellipse
        ellipse_image[
            : int(ellipse_image.shape[0] / 2 - 1), : int(ellipse_image.shape[1] / 2 - 1)
        ] = 0
        ellipse_image[
            -int(ellipse_image.shape[0] / 2 - 1) :,
            -int(ellipse_image.shape[1] / 2 - 1) :,
        ] = 0
        ellipses, _ = fitter.fit_ellipses(ellipse_image)
        assert len(ellipses) == 0

    def test_fitter_rejects_overly_eccentric_ellipses(self, one_easy_ellipse, fitter):
        ellipse_image, _ = one_easy_ellipse
        fitter.min_eccentricity = 0.9
        ellipses, _ = fitter.fit_ellipses(ellipse_image)
        assert len(ellipses) == 0

    def test_fitter_rejects_too_small_ellipses(self, fitter, one_easy_ellipse_image):
        fitter.min_ellipse_long_axis = 100
        ellipses, _ = fitter.fit_ellipses(one_easy_ellipse_image)
        assert len(ellipses) == 0

    def test_fitter_rejects_too_large_ellipses(self, fitter, one_easy_ellipse_image):
        fitter.max_ellipse_long_axis = 1
        ellipses, _ = fitter.fit_ellipses(one_easy_ellipse_image)
        assert len(ellipses) == 0

    def test_fitter_attributes_are_settable_at_instantiation(self):
        args = {
            "min_area_ratio": 0.3,
            "min_eccentricity": 0.6,
            "min_ellipse_long_axis": 10.0,
            "max_ellipse_long_axis": 30,
            "max_ellipse_contour_centre_dist": 11,
            "min_area": 26,
            "max_area": 10001,
        }
        ellipse_fitter = sectioning.EllipseFitter(**args)
        for arg in args.keys():
            if isinstance(args[arg], np.ndarray):
                np.testing.assert_array_equal(
                    ellipse_fitter.__getattribute__(arg), args[arg]
                )
            else:
                assert ellipse_fitter.__getattribute__(arg) == args[arg]


class TestCTEllipsoidFitter:
    @pytest.fixture()
    def fitter(self):
        return sectioning.CTEllipsoidFitter()

    @pytest.fixture()
    def ball(self):
        arr = phase_correlation_image_processing.sphere([50, 50, 50], radius=20)
        return np.fft.ifftshift(arr)

    @pytest.fixture()
    def small_ball(self):
        arr = phase_correlation_image_processing.sphere([50, 50, 50], radius=15)
        return np.fft.ifftshift(arr)

    @pytest.fixture()
    def ball_expected_walls(self, ball):
        zero_crossings_ball = phase_correlation_image_processing.zero_crossings(
            phase_correlation_image_processing.lmr(ball, radius=1)
        )
        return np.stack(zero_crossings_ball.nonzero(), axis=1)

    def test_walls_of_sphere_found(self, fitter, ball, ball_expected_walls):
        found_ellipsoid, _ = fitter.draw_ellipsoid_walls(ball)
        predicted_locations = np.stack((found_ellipsoid > 0).nonzero(), axis=1)
        dists = ball_expected_walls - predicted_locations[:, np.newaxis, :]
        dists = np.linalg.norm(dists, axis=-1).min(axis=1)
        assert (dists < 2).all()

    def test_walls_of_sphere_reach_three_at_right_locations(
        self, fitter, ball, ball_expected_walls
    ):
        found_ellipsoid, _ = fitter.draw_ellipsoid_walls(ball)
        predicted_locations = np.stack((found_ellipsoid == 3).nonzero(), axis=1)
        assert len(predicted_locations) > 0
        dists = ball_expected_walls - predicted_locations[:, np.newaxis, :]
        dists = np.linalg.norm(dists, axis=-1).min(axis=1)
        assert (dists < 2).all()

    def test_two_separate_axial_circles_dont_reach_two(self, fitter):
        arr = np.zeros([50, 50, 50])
        arr[5] = np.fft.ifftshift(
            phase_correlation_image_processing.circle([50, 50], 20)
        )
        arr[7] = np.fft.ifftshift(
            phase_correlation_image_processing.circle([50, 50], 20)
        )
        found_ellipsoid, _ = fitter.draw_ellipsoid_walls(arr)
        predicted_locations = np.stack((found_ellipsoid > 1).nonzero(), axis=1)
        assert len(predicted_locations) == 0

    @pytest.fixture()
    def medfilt2d_patch(self):
        medfilt2d_patch = mock.MagicMock()
        medfilt2d_patch.return_value = np.zeros([50, 50])
        return medfilt2d_patch

    @pytest.fixture()
    def sectioner_patch(self, monkeypatch):
        patch = mock.MagicMock()
        patch.label_im.return_value = np.zeros([50, 50])
        return patch

    @pytest.fixture()
    def dummy_sectioner_kwargs(self):
        return {"full_sub_structure": True, "threshold": 500}

    @pytest.fixture()
    def find_ellipsoids_patch(self, monkeypatch):
        monkeypatch.setattr(
            sectioning.CTEllipsoidFitter, "find_ellipsoids", mock.MagicMock()
        )

    def test_medfilt2d_used_if_set(
        self,
        ball,
        medfilt2d_patch,
        fitter,
        sectioner_patch,
        dummy_sectioner_kwargs,
        find_ellipsoids_patch,
    ):
        fitter.draw_ellipsoid_walls(
            ball,
            filterer=medfilt2d_patch,
            sectioner=sectioner_patch,
            sectioner_kwargs=dummy_sectioner_kwargs,
        )
        assert medfilt2d_patch.call_count > 0

    def test_sectioner_used_if_set(
        self,
        ball,
        fitter,
        sectioner_patch,
        dummy_sectioner_kwargs,
        find_ellipsoids_patch,
    ):
        fitter.draw_ellipsoid_walls(
            ball, sectioner=sectioner_patch, sectioner_kwargs=dummy_sectioner_kwargs
        )
        assert sectioner_patch.label_im.call_count > 0

    def test_ellipses_to_rich_information_correctly(self, fitter):
        ellipse_2d = [((1.0, 2.0), (3.0, 4.0), 5.0)]
        expected_area = 3 * 4 * np.pi / 4
        ellipse_3d = fitter._ellipses_to_rich_information(ellipse_2d, 0, 10, [11])
        assert ellipse_3d[0] == (
            (10.0, 2.0, 1.0),
            (3.0, 4.0),
            5.0,
            0,
            11,
            expected_area,
        )
        ellipse_3d = fitter._ellipses_to_rich_information(ellipse_2d, 1, 10, [11])
        assert ellipse_3d[0] == (
            (2.0, 10.0, 1.0),
            (3.0, 4.0),
            5.0,
            1,
            11,
            expected_area,
        )
        ellipse_3d = fitter._ellipses_to_rich_information(ellipse_2d, 2, 10, [11])
        assert ellipse_3d[0] == (
            (2.0, 1.0, 10.0),
            (3.0, 4.0),
            5.0,
            2,
            11,
            expected_area,
        )

    def test_nested_spheres_found(self, ball, small_ball, fitter):
        nested_balls = ball.astype("uint8") + small_ball
        assert nested_balls.max() == 2
        _, found_ellipsoids = fitter.draw_ellipsoid_walls(nested_balls)
        assert len(found_ellipsoids) >= 2
        for ellipsoid in found_ellipsoids:
            np.testing.assert_allclose(ellipsoid["centre"], (25, 25, 25))

    def test_flattened_ellipse_found(self, fitter):
        array_size = [50, 50, 50]
        y_start = np.ceil(-array_size[0] / 2)
        y_end = np.ceil(array_size[0] / 2)
        x_start = np.ceil(-array_size[1] / 2)
        x_end = np.ceil(array_size[1] / 2)
        z_start = np.ceil(-array_size[2] / 2)
        z_end = np.ceil(array_size[2] / 2)
        z, y, x = np.mgrid[z_start:z_end, y_start:y_end, x_start:x_end]
        # squash in z
        z *= 4
        ellipsoid = x ** 2 + y ** 2 + z ** 2 <= 20 ** 2
        _, found_ellipsoids = fitter.draw_ellipsoid_walls(ellipsoid)
        assert len(found_ellipsoids) > 0
        for ellipsoid in found_ellipsoids:
            np.testing.assert_allclose(ellipsoid["centre"], (25, 25, 25))

    def test_isolated_circle_not_found(self, fitter):
        arr = np.zeros([50, 50, 50])
        arr[25] = np.fft.ifftshift(
            phase_correlation_image_processing.circle([50, 50, 50], 15)
        )
        _, found_ellipsoids = fitter.draw_ellipsoid_walls(arr)
        assert len(found_ellipsoids) == 0


class TestDinoSectioner:
    @pytest.fixture()
    def patched_vit_tiny(self, monkeypatch):
        monkeypatch.setattr(sectioning.vits, "vit_tiny", mock.MagicMock())

    @pytest.fixture()
    def sectioner(self, patched_root_directory):
        return sectioning.DinoSectioner(total_samples=500, samples_per_image=500)

    @pytest.fixture()
    def sectioner_w_model(self, sectioner):
        sectioner.load_dino_model()
        return sectioner

    def test_load_model_uses_vits_w_correct_default(self, sectioner, patched_vit_tiny):
        sectioner.load_dino_model()
        expected_call = mock.call(patch_size=16, num_classes=0)
        assert sectioning.vits.vit_tiny.call_args == expected_call

    def test_image_to_attention_stack_gets_expected_output_shape(
        self, sectioner_w_model
    ):
        im = (np.random.rand(480, 480) * 255).astype("uint8")
        out_stack = sectioner_w_model.single_image_texton_descriptors(im)
        assert out_stack.shape == (3, *im.shape)

    def test_sectioner_can_build_sample_texton_set_and_train_clusterer(
        self, sectioner_w_model
    ):
        sectioner_w_model.build_sample_texton_set(threshold=0)
        assert sectioner_w_model.texton_sample_set.shape == (
            sectioner_w_model.total_samples,
            sectioner_w_model.total_filters,
        )
        sectioner_w_model.train_clusterers()
        im = (np.random.rand(480, 480) * 255).astype("uint8")
        assert isinstance(
            sectioner_w_model.label_im(im, 0, 0, full_sub_structure=True), np.ndarray
        )
