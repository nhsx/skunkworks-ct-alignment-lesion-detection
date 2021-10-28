from ai_ct_scans import phase_correlation
from skimage import draw
import pytest
import numpy as np
import mock


@pytest.fixture()
def circle():
    circle = np.zeros([100, 100], dtype=float)
    rows, cols = draw.circle(50, 50, 25)
    circle[rows, cols] = 1
    return circle


@pytest.fixture()
def shift_2d():
    return 5, 18


@pytest.fixture()
def shifted_circle(circle, shift_2d):
    shifted = np.roll(circle, shift_2d, axis=(0, 1))
    return shifted


def test_align_via_phase_correlation_2d_recovers_shifted_circle(circle, shifted_circle):
    unshifted_circle = phase_correlation.align_via_phase_correlation_2d(
        circle, shifted_circle
    )
    np.testing.assert_array_equal(circle, unshifted_circle)


@pytest.fixture()
def sphere():
    sphere_array_size = 50
    sphere_radius = 10
    x, y, z = np.mgrid[0:sphere_array_size, 0:sphere_array_size, 0:sphere_array_size]
    return ((x - 5) ** 2 + (y - 5) ** 2 + (z - 5) ** 2) < (sphere_radius ** 2)


@pytest.fixture()
def shift_3d():
    return 5, 18, 7


@pytest.fixture()
def shifted_sphere(sphere, shift_3d):
    shifted = np.roll(sphere, shift_3d, axis=(0, 1, 2))
    return shifted


def test_shift_via_phase_correlation_nd_gets_expected_shift_in_2d(
    circle, shifted_circle, shift_2d
):
    shifts = phase_correlation.shift_via_phase_correlation_nd([circle, shifted_circle])
    np.testing.assert_array_equal(shifts[1], list(shift_2d))


def test_shift_via_phase_correlation_nd_gets_expected_shift_in_3d(
    sphere, shifted_sphere, shift_3d
):
    shifts = phase_correlation.shift_via_phase_correlation_nd([sphere, shifted_sphere])
    np.testing.assert_array_equal(shifts[1], list(shift_3d))


def test_shift_via_phase_correlation_nd_deals_with_irregular_image_shapes(
    circle, shifted_circle, shift_2d
):
    shifted_cropped = shifted_circle[
        : shifted_circle.shape[0] - 1, : shifted_circle.shape[1] - 1
    ]
    shifts = phase_correlation.shift_via_phase_correlation_nd([circle, shifted_cropped])
    np.testing.assert_array_equal(shifts[1], list(shift_2d))


def test_shift_via_phase_correlation_nd_deals_with_negative_rolled_images(circle):
    shift = (-16, -5)
    shifted = np.roll(circle, shift, axis=(0, 1))
    shifts = phase_correlation.shift_via_phase_correlation_nd([circle, shifted])
    np.testing.assert_array_equal(shifts[1], np.array(shift))


@pytest.fixture()
def patched_lmr(monkeypatch):
    monkeypatch.setattr(
        phase_correlation.phase_correlation_image_processing, "lmr", mock.MagicMock()
    )


def test_lmr_not_used_if_apply_lmr_false(patched_lmr, ims_nonlinear_features_2d):
    ims, _, _, local_coords, region_widths = ims_nonlinear_features_2d
    phase_correlation.shifts_via_local_region(
        ims, local_coords=local_coords[0], region_widths=region_widths, apply_lmr=False
    )
    phase_correlation.phase_correlation_image_processing.lmr.assert_not_called()


@pytest.fixture()
def patched_zero_crossings(monkeypatch):
    monkeypatch.setattr(
        phase_correlation.phase_correlation_image_processing,
        "zero_crossings",
        mock.MagicMock(),
    )


def test_zero_crossings_not_used_if_apply_zero_crossings_false(
    patched_zero_crossings, ims_nonlinear_features_2d
):
    ims, _, _, local_coords, region_widths = ims_nonlinear_features_2d
    phase_correlation.shifts_via_local_region(
        ims,
        local_coords=local_coords[0],
        region_widths=region_widths,
        lmr_radius=5,
        apply_zero_crossings=False,
    )
    phase_correlation.phase_correlation_image_processing.zero_crossings.assert_not_called()


@pytest.fixture()
def ims_nonlinear_features_2d():
    ims = [np.zeros([150, 100]) for i in range(2)]
    feature_1_offsets = [3, 4]
    feature_2_offsets = [5, 6]
    feature_1_start_stop = [[10, 20], [10, 20]]
    ims[0][
        feature_1_start_stop[0][0] : feature_1_start_stop[0][1],
        feature_1_start_stop[1][0] : feature_1_start_stop[1][1],
    ] = 1
    ims[1][
        feature_1_start_stop[0][0]
        + feature_1_offsets[0] : feature_1_start_stop[0][1]
        + feature_1_offsets[0],
        feature_1_start_stop[1][0]
        + feature_1_offsets[1] : feature_1_start_stop[1][1]
        + feature_1_offsets[1],
    ] = 1
    feature_2_start_stop = [[100, 105], [60, 67]]
    ims[0][
        feature_2_start_stop[0][0] : feature_2_start_stop[0][1],
        feature_2_start_stop[1][0] : feature_2_start_stop[1][1],
    ] = 1
    ims[1][
        feature_2_start_stop[0][0]
        + feature_2_offsets[0] : feature_2_start_stop[0][1]
        + feature_2_offsets[0],
        feature_2_start_stop[1][0]
        + feature_2_offsets[1] : feature_2_start_stop[1][1]
        + feature_2_offsets[1],
    ] = 1
    feature_offsets = (feature_1_offsets, feature_2_offsets)
    feature_start_stops = (feature_1_start_stop, feature_2_start_stop)
    local_coords = (
        np.vstack(
            [np.array(feature_1_start_stop)[:, 0], np.array(feature_2_start_stop)[:, 0]]
        )
        + 5
    )
    region_widths = [15, 15]

    return ims, feature_offsets, feature_start_stops, local_coords, region_widths


@pytest.fixture()
def shifts_via_local_region_2d(ims_nonlinear_features_2d):
    ims, _, _, local_coords, region_widths = ims_nonlinear_features_2d

    shifts_1 = phase_correlation.shifts_via_local_region(
        ims, local_coords=local_coords[0], region_widths=region_widths, lmr_radius=3
    )
    shifts_2 = phase_correlation.shifts_via_local_region(
        ims, local_coords=local_coords[1], region_widths=region_widths, lmr_radius=3
    )
    return shifts_1, shifts_2


def test_shifts_via_local_region_gets_correct_shifts_2d(
    shifts_via_local_region_2d, ims_nonlinear_features_2d
):
    _, (feature_1_offsets, feature_2_offsets), _, _, _ = ims_nonlinear_features_2d
    shifts_1, shifts_2 = shifts_via_local_region_2d
    np.testing.assert_array_equal(shifts_1[1], feature_1_offsets)
    np.testing.assert_array_equal(shifts_2[1], feature_2_offsets)


@pytest.fixture()
def ims_nonlinear_features_3d():
    ims = [np.zeros([150, 100, 120]) for i in range(2)]
    feature_1_offsets = [3, 4, 5]
    feature_2_offsets = [5, 6, 7]
    feature_1_start_stop = [[10, 20], [10, 20], [14, 22]]
    ims[0][
        feature_1_start_stop[0][0] : feature_1_start_stop[0][1],
        feature_1_start_stop[1][0] : feature_1_start_stop[1][1],
        feature_1_start_stop[2][0] : feature_1_start_stop[2][1],
    ] = 1
    ims[1][
        feature_1_start_stop[0][0]
        + feature_1_offsets[0] : feature_1_start_stop[0][1]
        + feature_1_offsets[0],
        feature_1_start_stop[1][0]
        + feature_1_offsets[1] : feature_1_start_stop[1][1]
        + feature_1_offsets[1],
        feature_1_start_stop[2][0]
        + feature_1_offsets[2] : feature_1_start_stop[2][1]
        + feature_1_offsets[2],
    ] = 1
    feature_2_start_stop = [[100, 105], [60, 67], [25, 44]]
    ims[0][
        feature_2_start_stop[0][0] : feature_2_start_stop[0][1],
        feature_2_start_stop[1][0] : feature_2_start_stop[1][1],
        feature_2_start_stop[2][0] : feature_2_start_stop[2][1],
    ] = 1
    ims[1][
        feature_2_start_stop[0][0]
        + feature_2_offsets[0] : feature_2_start_stop[0][1]
        + feature_2_offsets[0],
        feature_2_start_stop[1][0]
        + feature_2_offsets[1] : feature_2_start_stop[1][1]
        + feature_2_offsets[1],
        feature_2_start_stop[2][0]
        + feature_2_offsets[2] : feature_2_start_stop[2][1]
        + feature_2_offsets[2],
    ] = 1
    feature_offsets = (feature_1_offsets, feature_2_offsets)
    feature_start_stops = (feature_1_start_stop, feature_2_start_stop)
    local_coords = (
        np.vstack(
            [np.array(feature_1_start_stop)[:, 0], np.array(feature_2_start_stop)[:, 0]]
        )
        + 5
    )
    region_widths = [15, 15]

    return ims, feature_offsets, feature_start_stops, local_coords, region_widths


@pytest.fixture()
def shifts_via_local_region_3d(ims_nonlinear_features_3d):
    ims, _, _, local_coords, region_widths = ims_nonlinear_features_3d

    shifts_1 = phase_correlation.shifts_via_local_region(
        ims, local_coords=local_coords[0], region_widths=region_widths, lmr_radius=3
    )
    shifts_2 = phase_correlation.shifts_via_local_region(
        ims, local_coords=local_coords[1], region_widths=region_widths, lmr_radius=3
    )
    return shifts_1, shifts_2


def test_shifts_via_local_region_gets_correct_shifts_3d(
    shifts_via_local_region_3d, ims_nonlinear_features_3d
):
    _, (feature_1_offsets, feature_2_offsets), _, _, _ = ims_nonlinear_features_3d
    shifts_1, shifts_2 = shifts_via_local_region_3d
    np.testing.assert_array_equal(shifts_1[1], feature_1_offsets)
    np.testing.assert_array_equal(shifts_2[1], feature_2_offsets)


@pytest.mark.parametrize("intensity", (0.001, 1000))
def test_shifts_via_local_region_on_extreme_image_intensities_with_noise_2d(
    ims_nonlinear_features_2d, intensity
):
    np.random.seed(532)
    (
        ims,
        (feature_1_offsets, feature_2_offsets),
        _,
        local_coords,
        region_widths,
    ) = ims_nonlinear_features_2d
    ims = [(im + (1 - 0.5 * np.random.rand(*im.shape))) * intensity for im in ims]

    shifts_1 = phase_correlation.shifts_via_local_region(
        ims,
        local_coords=local_coords[0],
        region_widths=region_widths,
        lmr_radius=3,
        zero_crossings_thresh="auto",
    )
    shifts_2 = phase_correlation.shifts_via_local_region(
        ims,
        local_coords=local_coords[1],
        region_widths=region_widths,
        lmr_radius=3,
        zero_crossings_thresh="auto",
    )
    np.testing.assert_array_equal(shifts_1[1], feature_1_offsets)
    np.testing.assert_array_equal(shifts_2[1], feature_2_offsets)


@pytest.mark.parametrize("intensity", (0.001, 1000))
def test_shifts_via_local_region_on_extreme_image_intensities_with_noise_3d(
    ims_nonlinear_features_3d, intensity
):
    np.random.seed(532)
    (
        ims,
        (feature_1_offsets, feature_2_offsets),
        _,
        local_coords,
        region_widths,
    ) = ims_nonlinear_features_3d
    ims = [(im + (1 - 0.5 * np.random.rand(*im.shape))) * intensity for im in ims]

    shifts_1 = phase_correlation.shifts_via_local_region(
        ims,
        local_coords=local_coords[0],
        region_widths=region_widths,
        lmr_radius=3,
        zero_crossings_thresh="auto",
    )
    shifts_2 = phase_correlation.shifts_via_local_region(
        ims,
        local_coords=local_coords[1],
        region_widths=region_widths,
        lmr_radius=3,
        zero_crossings_thresh="auto",
    )
    np.testing.assert_array_equal(shifts_1[1], feature_1_offsets)
    np.testing.assert_array_equal(shifts_2[1], feature_2_offsets)


@pytest.mark.parametrize("n_dims", (2, 3))
def test_shift_nd_fills_border_with_blanks(n_dims):
    shape = [15 for _ in range(n_dims)]
    shift = [3 for _ in range(n_dims)]
    arr = np.random.rand(*shape)
    shifted = phase_correlation.shift_nd(arr, shift)
    slices = [np.s_[0:shift_element] for shift_element in shift]
    assert (shifted[slices] == 0).all()


def test_shift_nd_deals_with_zero_shifts_and_negatives():
    shape = [15 for _ in range(3)]
    shift = [3, 0, -1]
    arr = np.random.rand(*shape)
    shifted = phase_correlation.shift_nd(arr, shift)
    assert (shifted[:3, :, :] == 0).all()
    assert (shifted[:, :, -1:] == 0).all()
    np.testing.assert_array_equal(shifted[3:, :, :-1], arr[:-3, :, 1:])
