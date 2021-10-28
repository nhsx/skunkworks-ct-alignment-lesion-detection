from ai_ct_scans import phase_correlation_image_processing
import numpy as np
import pytest


def test_zero_crossings_2d():
    im = np.ones([5, 5])
    im[2, 2] = -1
    crossings = phase_correlation_image_processing.zero_crossings(im)
    expected = np.zeros_like(im, dtype=bool)
    expected[1:3, 1:3] = 1
    np.testing.assert_array_equal(crossings, expected)


def test_zero_crossings_3d():
    im = np.ones([5, 5, 5])
    im[2, 2, 2] = -1
    crossings = phase_correlation_image_processing.zero_crossings(im)
    expected = np.zeros_like(im, dtype=bool)
    expected[1:3, 1:3, 1:3] = 1
    np.testing.assert_array_equal(crossings, expected)


def test_sphere_gets_expected_for_small_radius():
    array_shape = [5, 5, 5]
    out = phase_correlation_image_processing.sphere(array_shape, 1)
    expected = np.zeros(array_shape)
    expected[1:4, 2, 2] = 1
    expected[2, 1:4, 2] = 1
    expected[2, 2, 1:4] = 1
    expected = np.fft.ifftshift(expected)
    np.testing.assert_array_equal(out, expected)


def test_lmr_2d():
    im = np.zeros([10, 10])
    im[5, 5] = 1
    out_im = phase_correlation_image_processing.lmr(im, radius=3)
    assert out_im.mean() < im.mean()
    assert out_im[5, 5] < 1
    assert out_im[5, 4] < 0
    assert out_im.dtype == "float"
    assert out_im.shape == im.shape


def test_lmr_3d():
    im = np.zeros([10, 9, 8])
    im[5, 5, 5] = 1
    out_im = phase_correlation_image_processing.lmr(im, radius=3)
    assert out_im.mean() < im.mean()
    assert out_im[5, 5, 5] < 1
    assert out_im[5, 4, 5] < 0
    assert out_im.dtype == "float"
    assert out_im.shape == im.shape


def test_lmr_accepts_defined_filter():
    out = phase_correlation_image_processing.lmr(
        np.random.rand(5, 5), filter_type=np.random.rand(5, 5)
    )
    assert out.shape == (5, 5)


def test_no_filter_no_radius_raises_error_in_lmr():
    with pytest.raises(Exception):
        phase_correlation_image_processing.lmr(None, None, None)


@pytest.mark.parametrize("image_shape", ([3, 4], [5, 6]))
def test_generate_overlay_gets_correct_output(image_shape):
    np.random.seed(555)
    # get images with 1 pixel difference size in each dimension
    ims = [np.random.rand(*list(np.array(image_shape) + i)) for i in range(2)]
    overlay = phase_correlation_image_processing.generate_overlay_2d(
        ims, normalize=False
    )
    for im_i, im in enumerate(ims):
        np.testing.assert_array_equal(overlay[: im.shape[0], : im.shape[1], im_i], im)


def test_generate_overlay_gets_correct_output_normed():
    np.random.seed(555)
    # get images with 1 pixel difference size in each dimension
    ims = [np.random.rand(*list(np.array([3, 4]) + i)) for i in range(2)]
    overlay = phase_correlation_image_processing.generate_overlay_2d(ims)
    for im_i, im in enumerate(ims):
        np.testing.assert_array_equal(
            overlay[: im.shape[0], : im.shape[1], im_i], im / im.max()
        )


def test_pad_nd_gets_expected_output():
    arr = np.random.rand(3, 3)
    new_shape = (5, 6)
    out = phase_correlation_image_processing.pad_nd(arr, new_shape)
    assert out.shape == new_shape
    np.testing.assert_array_equal(out[:3, :3], arr)
    assert (out[3:] == out[:3, :3].mean()).all()
    assert (out[:, 3:] == out[:3, :3].mean()).all()
