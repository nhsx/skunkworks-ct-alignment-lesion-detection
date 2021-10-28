from pathlib import Path
import copy

import mock
import pytest
from ai_ct_scans import model_trainers
import numpy as np
import torch

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = "cpu"


class TestInfillTrainer:
    def test_infill_trainer_gets_expected_num_patients(self, infill_trainer):
        assert len(infill_trainer.multi_patient_loader.patients) == 2

    def test_infill_trainer_has_memmaps(self, infill_trainer):
        for patient in infill_trainer.multi_patient_loader.patients:
            assert isinstance(patient.abdo.scan_1.full_memmap, np.ndarray)
            assert isinstance(patient.abdo.scan_2.full_memmap, np.ndarray)
            # assert isinstance(patient.thorax.scan_1.full_memmap, np.ndarray)
            # assert isinstance(patient.thorax.scan_2.full_memmap, np.ndarray)

    @pytest.fixture()
    def patched_loss_masks(self, monkeypatch):
        patch = mock.MagicMock()
        patch.return_value = torch.Tensor(np.random.rand(128, 128, 128)).to(dev)
        monkeypatch.setattr(
            model_trainers.InfillTrainer, "_convert_loss_masks_to_tensor", patch
        )

    def test_random_axial_slice_gets_correct_shape(
        self, patched_loss_masks, patched_root_directory
    ):
        shallow_slices_trainer = model_trainers.InfillTrainer(2, 128, 128)
        out, _ = shallow_slices_trainer.random_axial_slicer(
            shallow_slices_trainer.multi_patient_loader.patients[
                0
            ].abdo.scan_1.full_memmap
        )
        assert out.shape == (
            shallow_slices_trainer.coronal_width,
            shallow_slices_trainer.sagittal_width,
        )

    def test_random_coronal_slice_gets_correct_output(
        self, infill_trainer, monkeypatch
    ):
        patch = mock.MagicMock()
        out_indices = np.array([100, 5, 120])
        patch.return_value = out_indices
        monkeypatch.setattr(model_trainers.np.random, "randint", patch)
        in_vol = np.random.rand(512, 10, 511)
        arr, indices = infill_trainer.random_coronal_slicer(in_vol)
        expected = in_vol[
            out_indices[0] : out_indices[0] + infill_trainer.axial_width,
            out_indices[1],
            out_indices[2] : out_indices[2] + infill_trainer.sagittal_width,
        ]
        assert arr.shape == (infill_trainer.axial_width, infill_trainer.sagittal_width)
        np.testing.assert_array_equal(expected, arr)
        np.testing.assert_array_equal(indices, out_indices)

    def test_random_sagittal_slice_gets_correct_output(
        self, infill_trainer, monkeypatch
    ):
        patch = mock.MagicMock()
        out_indices = np.array([100, 120, 5])
        patch.return_value = out_indices
        monkeypatch.setattr(model_trainers.np.random, "randint", patch)
        in_vol = np.random.rand(512, 511, 10)
        arr, indices = infill_trainer.random_sagittal_slicer(in_vol)
        expected = in_vol[
            out_indices[0] : out_indices[0] + infill_trainer.axial_width,
            out_indices[1] : out_indices[1] + infill_trainer.coronal_width,
            out_indices[2],
        ]
        assert arr.shape == (infill_trainer.axial_width, infill_trainer.coronal_width)
        np.testing.assert_array_equal(expected, arr)
        np.testing.assert_array_equal(indices, out_indices)

    def test_random_axial_slice_gets_correct_output(self, infill_trainer, monkeypatch):
        patch = mock.MagicMock()
        out_indices = np.array([5, 120, 100])
        patch.return_value = out_indices
        monkeypatch.setattr(model_trainers.np.random, "randint", patch)
        in_vol = np.random.rand(10, 511, 512)
        arr, indices = infill_trainer.random_axial_slicer(in_vol)
        expected = in_vol[
            out_indices[0],
            out_indices[1] : out_indices[1] + infill_trainer.coronal_width,
            out_indices[2] : out_indices[2] + infill_trainer.sagittal_width,
        ]
        assert arr.shape == (
            infill_trainer.coronal_width,
            infill_trainer.sagittal_width,
        )
        np.testing.assert_array_equal(expected, arr)
        np.testing.assert_array_equal(indices, out_indices)

    def test_plane_masks_makes_expected_masks(self, infill_trainer):
        assert infill_trainer.plane_masks[0].shape == (
            infill_trainer.coronal_width,
            infill_trainer.sagittal_width,
        )
        assert infill_trainer.plane_masks[1].shape == (
            infill_trainer.axial_width,
            infill_trainer.sagittal_width,
        )
        assert infill_trainer.plane_masks[2].shape == (
            infill_trainer.axial_width,
            infill_trainer.coronal_width,
        )
        for mask in infill_trainer.plane_masks:
            row_start = int(
                np.floor(mask.shape[0] / 2) - infill_trainer.blank_width / 2
            )
            col_start = int(
                np.floor(mask.shape[1] / 2) - infill_trainer.blank_width / 2
            )
            assert (
                mask[
                    row_start : row_start + infill_trainer.blank_width,
                    col_start : col_start + infill_trainer.blank_width,
                ]
                == 0
            ).all()
            assert (mask[:row_start] == 1).all()
            assert (mask[row_start + infill_trainer.blank_width :] == 1).all()
            assert (mask[:, :col_start] == 1).all()
            assert (mask[:, col_start + infill_trainer.blank_width] == 1).all()

    def test_build_batch_builds_valid_batch(
        self, patched_batch, patched_infill_trainer_slicers
    ):
        assert isinstance(patched_batch["input_images"], torch.Tensor)
        assert isinstance(patched_batch["input_planes"], torch.Tensor)
        assert isinstance(patched_batch["input_body_part"], torch.Tensor)
        assert isinstance(patched_batch["input_coords"], torch.Tensor)
        assert isinstance(patched_batch["labels"], torch.Tensor)
        assert patched_batch["labels"].shape == (
            patched_infill_trainer_slicers.batch_size,
            1,
            patched_infill_trainer_slicers.coronal_width,
            patched_infill_trainer_slicers.sagittal_width,
        )
        assert patched_batch["input_images"].shape == (
            patched_infill_trainer_slicers.batch_size,
            1,
            patched_infill_trainer_slicers.batch_height,
            patched_infill_trainer_slicers.batch_width,
        )
        assert patched_batch["input_planes"].shape == (
            patched_infill_trainer_slicers.batch_size,
            3,
        )
        assert patched_batch["input_body_part"].shape == (
            patched_infill_trainer_slicers.batch_size,
            2,
        )
        assert patched_batch["input_coords"].shape == (
            patched_infill_trainer_slicers.batch_size,
            3,
        )

    def test_train_step_changes_weights(self, patched_infill_trainer_slicers):
        start_weights_first_layer = np.copy(
            patched_infill_trainer_slicers.model.state_dict()["encoder_convs.0.weight"]
            .cpu()
            .detach()
            .numpy()[:10, :10]
        )
        patched_infill_trainer_slicers.train_step()
        end_weights_first_layer = (
            patched_infill_trainer_slicers.model.state_dict()["encoder_convs.0.weight"]
            .cpu()
            .detach()
            .numpy()[:10, :10]
        )
        assert not (start_weights_first_layer == end_weights_first_layer).all()

    def test_running_loss_increments_length_if_not_filled(
        self, patched_infill_trainer_slicers
    ):
        patched_infill_trainer_slicers.last_n_losses = [
            i for i in range(patched_infill_trainer_slicers.loss_num_to_ave_over - 1)
        ]
        start_length = len(patched_infill_trainer_slicers.last_n_losses)
        patched_infill_trainer_slicers.train_step()
        end_length = len(patched_infill_trainer_slicers.last_n_losses)
        assert (start_length + 1) == end_length

    def test_running_loss_stays_set_length_if_already_filled(
        self, patched_infill_trainer_slicers
    ):
        patched_infill_trainer_slicers.last_n_losses = [
            i for i in range(patched_infill_trainer_slicers.loss_num_to_ave_over)
        ]
        patched_infill_trainer_slicers.train_step()
        assert (
            len(patched_infill_trainer_slicers.last_n_losses)
            == patched_infill_trainer_slicers.loss_num_to_ave_over
        )

    def test_train_step_increments_iteration(self, patched_infill_trainer_slicers):
        start_iteration = patched_infill_trainer_slicers.iteration
        patched_infill_trainer_slicers.train_step()
        assert (start_iteration + 1) == patched_infill_trainer_slicers.iteration

    def test_save_model_saves_model_with_same_weights(
        self, patched_infill_trainer_slicers, tmpdir
    ):
        # tmpdir needs converting because it doesn't behave quite like a pathlib path
        save_dir = Path(tmpdir)
        # to get a valid loss to save with, run a train step
        patched_infill_trainer_slicers.train_step()
        start_weights = np.copy(
            patched_infill_trainer_slicers.model.state_dict()["encoder_convs.0.weight"]
            .cpu()
            .detach()
            .numpy()[:10, :10]
        )
        patched_infill_trainer_slicers.save_model(save_dir, bypass_loss_check=True)
        expected_running_loss = copy.deepcopy(
            patched_infill_trainer_slicers.last_n_losses
        )
        # cause weights to become different in the model
        patched_infill_trainer_slicers.train_step()
        different_weights = np.copy(
            patched_infill_trainer_slicers.model.state_dict()["encoder_convs.0.weight"]
            .cpu()
            .detach()
            .numpy()[:10, :10]
        )
        # recover original weights by reloading
        patched_infill_trainer_slicers.load_model(save_dir)
        end_weights = (
            patched_infill_trainer_slicers.model.state_dict()["encoder_convs.0.weight"]
            .cpu()
            .detach()
            .numpy()[:10, :10]
        )
        assert not (start_weights == different_weights).all()
        np.testing.assert_array_equal(start_weights, end_weights)
        # check iterations were reset to saved state
        assert patched_infill_trainer_slicers.iteration == 1
        np.testing.assert_array_equal(
            patched_infill_trainer_slicers.last_n_losses, expected_running_loss
        )

    def test_save_used_expected_num_times_during_train_for_iterations(
        self, patched_infill_trainer_slicers, monkeypatch
    ):
        monkeypatch.setattr(
            patched_infill_trainer_slicers, "save_model", mock.MagicMock()
        )
        patched_infill_trainer_slicers.save_freq = 2
        patched_infill_trainer_slicers.train_for_iterations(4)
        assert patched_infill_trainer_slicers.save_model.call_count == 2

    def test_delete_memmap_is_called_if_clear_previous_memmaps_true(
        self, infill_trainer_del_memmap_true, patched_delete_memmap
    ):
        assert (
            infill_trainer_del_memmap_true.multi_patient_loader.patients[
                0
            ].abdo.scan_1.delete_memmap.call_count
            > 0
        )
