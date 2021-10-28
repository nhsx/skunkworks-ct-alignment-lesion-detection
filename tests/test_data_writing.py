import numpy as np
import pydicom
from ai_ct_scans import data_writing
from ai_ct_scans.data_loading import load_memmap


def test_create_dicom_file_writes_correct_array(tmp_path):
    file_path = str(tmp_path / "I0")
    dummy_array = np.random.rand(512, 512) * 255
    expected = dummy_array.astype("uint16")
    data_writing.create_dicom_file(file_path, dummy_array)
    reloaded_dicom = pydicom.read_file(file_path)
    np.testing.assert_array_equal(reloaded_dicom.pixel_array, expected)


def test_ndarray_to_memmap_writes_loadable_memmap(tmp_path):
    filename = "dummy_memmap.npy"
    arr = np.random.rand(3, 2)
    path = data_writing.ndarray_to_memmap(arr, tmp_path, filename)
    loaded = load_memmap(path)
    np.testing.assert_array_equal(arr, loaded)


def test_ndarray_to_memmap_handles_non_float64(tmp_path):
    filename = "dummy_memmap.npy"
    dtype = bool
    arr = np.ones([3, 2], dtype=dtype)
    path = data_writing.ndarray_to_memmap(arr, tmp_path, filename)
    loaded = load_memmap(path, dtype=dtype)
    np.testing.assert_array_equal(arr, loaded)
