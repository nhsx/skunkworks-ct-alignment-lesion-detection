from ai_ct_scans import models
import pytest


@pytest.fixture()
def infiller(patched_batch, patched_infill_trainer_slicers):
    return models.Infiller()


def test_infiller_processes_batch_to_expected_tensor_shape(
    infiller, patched_batch, patched_infill_trainer_slicers
):
    out = infiller(patched_batch)
    expected_shape = (
        patched_infill_trainer_slicers.batch_size,
        1,
        patched_infill_trainer_slicers.blank_width,
        patched_infill_trainer_slicers.blank_width,
    )
    out.shape == expected_shape
