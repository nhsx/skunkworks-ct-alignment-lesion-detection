import numpy as np
import pytest

from ai_ct_scans import image_processing_utils


class TestNormalisation:
    @pytest.mark.parametrize("shape", ((1, 2), (5, 5), (100, 10), (512, 512)))
    def test_same_shape_image_output(self, shape):
        output = image_processing_utils.normalise(np.ones(shape))
        assert output.shape == shape

    def test_output_is_8bit_image(self):
        output = image_processing_utils.normalise(np.ones((10, 10), dtype=np.uint16))
        assert output.dtype == np.uint8

    def test_on_16bit_image(self):
        test_image = np.zeros((9, 9))
        test_image[:, 3:6] = 1000
        test_image[:, 6:] = 2000

        expected = np.zeros((9, 9))
        expected[:, 3:6] = 127
        expected[:, 6:] = 255

        np.testing.assert_array_equal(
            image_processing_utils.normalise(test_image), expected
        )
