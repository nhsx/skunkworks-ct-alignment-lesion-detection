"""
Script to load the first trained model (by running first_try_training.py) and view the quality of infilling
"""

from ai_ct_scans.model_trainers import InfillTrainer
import torch
import numpy as np
import matplotlib.pyplot as plt
from ai_ct_scans.data_loading import data_root_directory

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = "cpu"

save_dir = data_root_directory().parent / "infiller_debug"

trainer = InfillTrainer(
    num_encoder_convs=6,
    num_decoder_convs=6,
    batch_size=3,
    clear_previous_memmaps=False,
    save_dir=save_dir,
    save_freq=1000,
    encoder_filts_per_layer=24,
    decoder_filts_per_layer=24,
    num_dense_layers=1,
    neurons_per_dense=128,
    learning_rate=0.00001,
    blank_width=16,
    kernel_size=7,
)
trainer.load_model(trainer.save_dir)

scan = trainer.multi_patient_loader.patients[0].abdo.scan_1.full_memmap

step = trainer.blank_width
window = 256

y_start = 100
x_start = 75
mid_layer = int(scan.shape[1] / 2)

im_stack = np.zeros([9, 1, window, window])
input_stack = np.zeros([9, 1, window, window])
labels = []
plane_one_hots = []
body_part_one_hots = []
coords_sets = []
input_mask = trainer.plane_masks[0]
label_mask = trainer.label_masks[0]

i = 0
for row_step in range(3):
    for col_step in range(3):
        y_offset = row_step * step
        x_offset = col_step * step
        slice = scan[
            y_start + y_offset : y_start + y_offset + window,
            mid_layer,
            x_start + x_offset : x_start + x_offset + window,
        ]
        label = slice[label_mask].reshape([1, step, step])
        labels.append(label)
        im_stack[i, 0, :, :] = slice
        input_stack[i, 0, :, :] = slice * input_mask
        plane_one_hots.append(np.array([0, 1, 0]))
        body_part_one_hots.append(np.array([1, 0]))
        coords_sets.append(
            np.array([y_start + y_offset, mid_layer, x_start + x_offset])
        )
        i += 1
labels = np.stack(labels)
in_dict = {
    "input_images": torch.Tensor(input_stack).to(dev),
    "input_planes": torch.Tensor(np.stack(plane_one_hots)).to(dev),
    "input_body_part": torch.Tensor(np.stack(body_part_one_hots)).to(dev),
    "input_coords": torch.Tensor(np.stack(coords_sets)).to(dev),
    "labels": torch.Tensor(labels).to(dev),
}

trainer.model.eval()
out = trainer.model(in_dict).cpu().detach().numpy()

f, axes = plt.subplots(3, 3, figsize=[16, 10])
axes = np.ravel(axes)
for i, ax in enumerate(axes):
    ax.imshow(im_stack[i, 0, :, :])
f.suptitle("Original input slices")
curr_path = save_dir / "original_slices.png"
plt.tight_layout()
plt.savefig(curr_path)

f, axes = plt.subplots(3, 3, figsize=[16, 10])
axes = np.ravel(axes)
for i, ax in enumerate(axes):
    ax.imshow(input_stack[i, 0, :, :])
f.suptitle("Masked slices")
curr_path = save_dir / "masked_slices.png"
plt.tight_layout()
plt.savefig(curr_path)

f, axes = plt.subplots(3, 3, figsize=[16, 10])
axes = np.ravel(axes)
for i, ax in enumerate(axes):
    ax.imshow(out[i, 0, :, :])
f.suptitle("Output slices")
curr_path = save_dir / "infilled_slices.png"
plt.tight_layout()
plt.savefig(curr_path)

f, axes = plt.subplots(3, 3, figsize=[16, 10])
axes = np.ravel(axes)
for i, ax in enumerate(axes):
    im = np.copy(input_stack[i, 0])
    # im[label_mask] -= out[i].reshape(-1)
    ax.imshow(out[i, 0] - im_stack[i, 0])
f.suptitle("Difference between original and infilled slices")
curr_path = save_dir / "difference_slices.png"
plt.tight_layout()
plt.savefig(curr_path)
