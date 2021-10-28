from pathlib import Path

from ai_ct_scans import model_trainers
import pytest
import mock
import numpy as np


@pytest.fixture()
def patched_root_directory(monkeypatch):
    patch = mock.MagicMock()
    patch.return_value = Path(__file__).parent / "fixtures" / "dicom_data"
    monkeypatch.setattr(model_trainers.data_loading, "data_root_directory", patch)


@pytest.fixture()
def infill_trainer(patched_root_directory):
    return model_trainers.InfillTrainer()


@pytest.fixture()
def patched_delete_memmap(monkeypatch):
    monkeypatch.setattr(
        model_trainers.data_loading.ScanLoader, "delete_memmap", mock.MagicMock()
    )


@pytest.fixture()
def infill_trainer_del_memmap_true(patched_root_directory, patched_delete_memmap):
    return model_trainers.InfillTrainer(clear_previous_memmaps=True)


@pytest.fixture()
def patched_infill_trainer_slicers(infill_trainer, patched_root_directory, monkeypatch):
    patched_axial_slicer = mock.MagicMock()
    patched_coronal_slicer = mock.MagicMock()
    patched_sagittal_slicer = mock.MagicMock()
    patched_axial_slicer.return_value = (
        np.random.rand(infill_trainer.coronal_width, infill_trainer.sagittal_width),
        np.random.randint(0, 3, 3),
    )
    patched_coronal_slicer.return_value = (
        np.random.rand(infill_trainer.axial_width, infill_trainer.sagittal_width),
        np.random.randint(0, 3, 3),
    )
    patched_sagittal_slicer.return_value = (
        np.random.rand(infill_trainer.axial_width, infill_trainer.coronal_width),
        np.random.randint(0, 3, 3),
    )
    monkeypatch.setattr(
        model_trainers.InfillTrainer, "random_axial_slicer", patched_axial_slicer
    )
    monkeypatch.setattr(
        model_trainers.InfillTrainer, "random_coronal_slicer", patched_coronal_slicer
    )
    monkeypatch.setattr(
        model_trainers.InfillTrainer, "random_sagittal_slicer", patched_sagittal_slicer
    )
    return model_trainers.InfillTrainer(learning_rate=1e-2)


@pytest.fixture()
def patched_batch(patched_infill_trainer_slicers):
    np.random.seed(555)
    return patched_infill_trainer_slicers.build_batch()
