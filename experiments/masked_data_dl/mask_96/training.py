"""
Experiment to train an infiller model with reasonably large (96x96 pixel) masked region at the centre.
"""


from ai_ct_scans.model_trainers import InfillTrainer
from ai_ct_scans.data_loading import data_root_directory
import numpy as np

np.random.seed(555)

save_dir = data_root_directory().parent / "infiller_with_blur_96_lr_1_4_mask"

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
    blank_width=96,
    kernel_size=7,
    blur_kernel=(5, 5),
    show_outline=True,
)
# leave this next line in to reload model from checkpoint
trainer.load_model(save_dir, model="latest_model.pth")
trainer.train_for_iterations(100000)
