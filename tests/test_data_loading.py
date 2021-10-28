from pathlib import Path

import pytest
from ai_ct_scans import data_loading
import pydicom
import mock
import numpy as np


@pytest.fixture()
def patient_1_dicom_dir():
    return Path(__file__).parent / "fixtures" / "dicom_data" / "1"


@pytest.fixture()
def patched_root_directory(monkeypatch):
    patch = mock.MagicMock()
    patch.return_value = Path(__file__).parent / "fixtures" / "dicom_data"
    monkeypatch.setattr(data_loading, "data_root_directory", patch)


@pytest.fixture()
def dicom_paths(patient_1_dicom_dir, patched_root_directory):
    return data_loading.dir_dicom_paths(patient_1_dicom_dir)


def test_paths_retrieved(dicom_paths):
    assert len(dicom_paths) > 0
    for path in dicom_paths:
        assert "I" in path.stem


def test_dicom_can_be_read(dicom_paths):
    for path in dicom_paths:
        out = pydicom.dcmread(path)
        assert out.pixel_array.shape == (384, 512)


def test_dicom_anonymisation(dicom_paths):
    for path in dicom_paths:
        out = pydicom.dcmread(path)
        assert "NAME^NONE" in out.PatientName


def test_data_root_directory_returns_expected_if_local_dir_exists():
    expected = Path(__file__).parents[1] / "extra_data" / "data"
    out = data_loading.data_root_directory()
    assert expected == out


@pytest.fixture()
def first_patient_dir(patched_root_directory):
    return data_loading.data_root_directory() / "1"


@pytest.fixture()
def pat_1_abdo_1_scan_loader(first_patient_dir):
    root_dir = first_patient_dir / "1" / "Abdo1"
    return data_loading.ScanLoader(root_dir)


def test_scan_loader_gets_expected_dicom_paths(pat_1_abdo_1_scan_loader):
    assert len(pat_1_abdo_1_scan_loader.dicom_paths) == 2
    for i in range(2):
        assert pat_1_abdo_1_scan_loader.dicom_paths[i].stem == f"I{i}"


def test_load_2d_array_gets_expected_ndarray_shape(pat_1_abdo_1_scan_loader):
    assert pat_1_abdo_1_scan_loader.load_2d_array(1, ignore_rescale=True).shape == (
        384,
        512,
    )


@pytest.fixture()
def loaded_abdo(pat_1_abdo_1_scan_loader):
    pat_1_abdo_1_scan_loader.load_scan()
    return pat_1_abdo_1_scan_loader


def test_load_scan_gets_expected_shape(loaded_abdo):
    assert loaded_abdo.full_scan.shape == (2, 192, 384)


@pytest.fixture()
def patched_mean_axial_thickness(monkeypatch):
    patch = mock.MagicMock()
    patch.return_value = 1.4
    monkeypatch.setattr(data_loading.ScanLoader, "mean_axial_thickness", patch)


def test_load_scan_gets_expected_shape_abnormal_axial_thickness(
    pat_1_abdo_1_scan_loader, patched_mean_axial_thickness
):
    pat_1_abdo_1_scan_loader.load_scan()
    assert pat_1_abdo_1_scan_loader.full_scan.shape == (4, 192, 384)


def test_load_scan_no_rescale_gets_expected_shape(first_patient_dir):
    root_dir = first_patient_dir / "1" / "Abdo1"
    abdo_scan_loader = data_loading.ScanLoader(root_dir, rescale_on_load=False)
    abdo_scan_loader.load_scan()
    assert abdo_scan_loader.full_scan.shape == (2, 384, 512)


def test_exception_raised_if_directory_does_not_exist(patched_root_directory):
    root_dir = (
        Path(__file__).parent
        / "fixtures"
        / "dicom_data_no_scan_1"
        / "1"
        / "1"
        / "Abdo1"
    )
    with pytest.raises(FileNotFoundError):
        data_loading.ScanLoader(root_dir=root_dir)


@pytest.fixture()
def abdo_loader(first_patient_dir):
    return data_loading.BodyPartLoader(root_dir=first_patient_dir, body_part="Abdo")


def test_abdo_loader_gets_expected_dicom_paths(abdo_loader):
    assert len(abdo_loader.scan_1.dicom_paths) == 2
    assert len(abdo_loader.scan_2.dicom_paths) == 2
    for i in range(2):
        assert abdo_loader.scan_1.dicom_paths[i].stem == f"I{i}"
    for i in range(2):
        assert abdo_loader.scan_2.dicom_paths[i].stem == f"I{i}"


def test_exception_raised_if_scan_1_does_not_exist(patched_root_directory):
    root_dir = Path(__file__).parent / "fixtures" / "dicom_data_no_scan_1" / "1"
    with pytest.raises(FileNotFoundError):
        data_loading.BodyPartLoader(root_dir=root_dir, body_part="Abdo")


@pytest.fixture()
def patient_loader(patched_root_directory):
    root_dir = data_loading.data_root_directory() / "1"
    return data_loading.PatientLoader(root_dir=root_dir)


# No need to test both scans as defaulted to one body type for simplicity
# def test_patientloader_gets_abdo_and_thorax_both_scans(patient_loader):
#     assert patient_loader.abdo.scan_1.dicom_paths[0].stem == 'I0'
#     assert patient_loader.thorax.scan_1.dicom_paths[0].stem == 'I0'


@pytest.fixture()
def multi_patient_loader(patched_root_directory):
    return data_loading.MultiPatientLoader()


def test_multipatientloader_gets_expected_patient_paths(multi_patient_loader):
    assert multi_patient_loader.patients[0].abdo.scan_1.dicom_paths[0].stem == "I0"
    assert multi_patient_loader.patients[1].abdo.scan_1.dicom_paths[0].stem == "I0"


def test_multipatientloader_with_alt_path(tmpdir):
    loader = data_loading.MultiPatientLoader(Path(tmpdir))
    assert loader.root_dir == Path(tmpdir)


def test_scan_loader_has_float_ndarray_pixel_spacing_after_load_scan(loaded_abdo):
    assert len(loaded_abdo.pixel_spacing) == 2
    for pixel_space in loaded_abdo.pixel_spacing:
        assert isinstance(pixel_space, float)


class TestMemmapping:
    """A class for testing the memory mapping of 3D arrays, and conversion to tensors for the DL pipeline"""

    @pytest.fixture()
    def abdo_written_memmap(self, loaded_abdo):
        loaded_abdo.full_scan_to_memmap()
        return loaded_abdo

    @pytest.fixture()
    def abdo_written_memmap_no_normalise(self, loaded_abdo):
        loaded_abdo.full_scan_to_memmap(normalise=False)
        return loaded_abdo

    @pytest.fixture()
    def abdo_loaded_memmap_no_normalise(self, loaded_abdo):
        loaded_abdo.load_memmap_and_clear_scan(normalise=False)
        return loaded_abdo

    def test_scan_loader_can_write_loadable_memmap(
        self, abdo_written_memmap_no_normalise
    ):
        abdo_written_memmap_no_normalise.load_full_memmap()
        np.testing.assert_array_equal(
            abdo_written_memmap_no_normalise.full_scan,
            abdo_written_memmap_no_normalise.full_memmap,
        )

    def test_scan_loader_can_find_memmap_if_not_explicitly_written(
        self, abdo_written_memmap_no_normalise
    ):
        abdo_written_memmap_no_normalise.memmap_save_path = None
        abdo_written_memmap_no_normalise.full_memmap = None
        abdo_written_memmap_no_normalise.load_full_memmap()
        np.testing.assert_array_equal(
            abdo_written_memmap_no_normalise.full_scan,
            abdo_written_memmap_no_normalise.full_memmap,
        )

    def test_find_memmap_returns_none_if_no_memmaps(
        self, abdo_written_memmap, monkeypatch
    ):
        patch = mock.MagicMock()
        patch.return_value = []
        monkeypatch.setattr(abdo_written_memmap, "_npy_list", patch)
        assert abdo_written_memmap._find_memmaps() == []

    def test_load_full_memmap_writes_memmap_if_none_found(
        self, abdo_written_memmap, monkeypatch
    ):
        abdo_written_memmap.memmap_save_paths = None
        patch_find_memmap = mock.MagicMock()
        patch_get_memmap_shape = mock.MagicMock()
        patch_get_memmap_shape.return_value = [1, 2, 3]
        monkeypatch.setattr(
            abdo_written_memmap, "_get_memmap_shape", patch_get_memmap_shape
        )
        monkeypatch.setattr(abdo_written_memmap, "_find_memmaps", patch_find_memmap)
        monkeypatch.setattr(
            abdo_written_memmap, "full_scan_to_memmap", mock.MagicMock()
        )
        monkeypatch.setattr(abdo_written_memmap, "load_scan", mock.MagicMock())
        monkeypatch.setattr(data_loading.np, "memmap", mock.MagicMock())
        abdo_written_memmap.load_full_memmap()
        abdo_written_memmap.full_scan_to_memmap.assert_called_once()
        abdo_written_memmap.load_scan.assert_called_once()

    def test_clear_full_scan_removes_full_scan_from_memory(self, abdo_written_memmap):
        abdo_written_memmap.clear_scan()
        assert abdo_written_memmap.full_scan is None

    @pytest.fixture()
    def abdo_loaded_memmap(self, abdo_written_memmap):
        abdo_written_memmap.load_memmap_and_clear_scan()
        return abdo_written_memmap

    def test_load_memmap_and_clear_scan_clears_full_scan(self, abdo_loaded_memmap):
        assert abdo_loaded_memmap.full_scan is None

    def test_memmap_normalised_by_default(self, abdo_loaded_memmap):
        assert abdo_loaded_memmap.full_memmap.min() >= 0.0
        assert abdo_loaded_memmap.full_memmap.max() <= 1.0

    def test_delete_memmap_deletes_memmap(self, abdo_written_memmap, monkeypatch):
        monkeypatch.setattr(data_loading.os, "remove", mock.MagicMock())
        abdo_written_memmap.delete_memmap()
        assert data_loading.os.remove.call_count > 0


class TestMultiPatientAxialStreamer:
    @pytest.fixture()
    def streamer(self, patched_root_directory):
        return data_loading.MultiPatientAxialStreamer()

    def test_stream_next_gets_expected_shape(self, streamer):
        out = streamer.stream_next()
        assert isinstance(out, np.ndarray)
        assert out.shape == (192, 384)

    def test_stream_next_twice_gets_different_outputs(self, streamer):
        out = [streamer.stream_next() for _ in range(2)]
        assert not (out[0] == out[1]).all()

    def test_stream_next_thrice_goes_to_second_scan(self, streamer):
        out = [streamer.stream_next() for _ in range(3)]
        for im in out[1:]:
            assert not (out[0] == im).all()

    def test_threshold_ensures_a_minimum_maximum(self, streamer):
        threshold = 309
        out = [streamer.stream_next(threshold=threshold) for _ in range(2)]
        for im in out:
            assert im.max() >= threshold

    @pytest.fixture()
    def patch_randint(self, monkeypatch):
        patch = mock.MagicMock()
        patch.return_value = 1
        monkeypatch.setattr(data_loading.np.random, "randint", patch)

    # def test_rand_used_to_select_if_random_true(self, streamer, patch_randint):
    #     streamer.stream_next(random=True)
    #     assert data_loading.np.random.randint.call_count > 0

    def test_reset_indices_sets_to_zero(self, streamer):
        streamer.stream_next(random=True)
        streamer.reset_indices()
        assert streamer.curr_patient_index == 0
        assert streamer.curr_body_part_index == 0
        assert streamer.curr_scan_num_index == 0
        assert streamer.curr_axial_index == 0
