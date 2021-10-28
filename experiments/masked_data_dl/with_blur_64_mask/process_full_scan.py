"""
An experiment for stitching the whole of a 3D scan together from only predicted regions of masked infiller output.
Saves the final 3D data to a .npy file for further experiments with normalisation and display in anomaly_renormalising.py
"""

from ai_ct_scans.model_trainers import InfillTrainer
import torch
from ai_ct_scans.data_loading import data_root_directory

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = "cpu"

save_dir = data_root_directory().parent / "infiller_with_blur_64_mask"

trainer = InfillTrainer(
    num_encoder_convs=6,
    num_decoder_convs=6,
    batch_size=9,
    clear_previous_memmaps=False,
    save_dir=save_dir,
    save_freq=1000,
    encoder_filts_per_layer=24,
    decoder_filts_per_layer=24,
    num_dense_layers=1,
    neurons_per_dense=128,
    learning_rate=0.000005,
    blank_width=64,
    kernel_size=7,
    blur_kernel=(5, 5),
    show_outline=True,
)
trainer.load_model(trainer.save_dir, "latest_model.pth")
trainer.process_full_scan(
    1,
    0,
    0,
    0,
    28,
    1,
    save_path=save_dir / "patient_2_anomaly_test",
    allow_off_edge=True,
)
