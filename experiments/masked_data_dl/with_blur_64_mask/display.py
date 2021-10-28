"""
Display output of an infiller model where a 64x64 masked region was present in input images, and a (5, 5) Gaussian
blur is applied to each image to avoid influence of noise in input. Edge information within the masked region
was also introduced in this experiment, which avoids the model failing due to normal but random structures within
the masked region
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
    learning_rate=0.00003,
    blank_width=64,
    kernel_size=7,
    blur_kernel=(5, 5),
    show_outline=True,
)
trainer.load_model(trainer.save_dir, "latest_model.pth")
trainer.model.eval()
torch.no_grad()

np.random.seed(567)
in_dict = trainer.build_batch()
out = trainer.model(in_dict).cpu().detach().numpy()
input_stack = in_dict["input_images"].cpu().detach().numpy()
im_stack = in_dict["labels"].cpu().detach().numpy()

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
diff_stack = np.abs(out - im_stack)
for i, ax in enumerate(axes):
    im = np.copy(input_stack[i, 0])
    # im[label_mask] -= out[i].reshape(-1)
    ax.imshow(diff_stack[i, 0])
f.suptitle("Difference between original and infilled slices")
curr_path = save_dir / "difference_slices.png"
plt.tight_layout()
plt.savefig(curr_path)

f, axes = plt.subplots(3, 3, figsize=[16, 10])
axes = np.ravel(axes)
vmin = 0
vmax = min(diff_stack.max(), 1)
for i, ax in enumerate(axes):
    im = np.copy(input_stack[i, 0])
    # im[label_mask] -= out[i].reshape(-1)
    curr_axis = ax.imshow(diff_stack[i, 0], vmin=vmin, vmax=vmax)

f.suptitle("Difference between original and infilled slices")
curr_path = save_dir / "difference_same_scale_slices.png"
plt.tight_layout()
# f.subplots_adjust(right=0.2)
f.colorbar(curr_axis, ax=axes.tolist())
plt.savefig(curr_path)
