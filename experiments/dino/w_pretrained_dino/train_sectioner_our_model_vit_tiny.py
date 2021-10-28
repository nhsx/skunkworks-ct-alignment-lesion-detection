from ai_ct_scans import sectioning
import numpy as np
import matplotlib.pyplot as plt

from ai_ct_scans.data_loading import data_root_directory

"""
Air is typically below an intensity of 500 and discarded for the purposes of training tissue sectioners as it otherwise
dominates the taken samples. Sectioned slices are also typically masked below this threshold to a -1 class. In
images sectioned using DINO output, this can lead to a higher resolution appearance of sectioning that is actually
achieved from DINO attention heads, at the boundary between air and bodily tissues.

The total samples correspond to the number of pixels for which a set of attention head outputs are used as feature
vectors, i.e. with 5000 total samples there will be 5000 data points, with the dataset taking the shape 
[5000, number of DINO model attention heads]. No qualitative improvement in sectioning was observed when increasing
this to 10000, sectioning models trained in a few minutes with 5000 data points, and so this value was accepted
"""
threshold = 500
total_samples = 5000
filter_type = ["custom_dino"]
arch = "vit_tiny"
patch_size = 5

dino_model_path = ""

sectioner = sectioning.DinoSectioner(total_samples=total_samples, samples_per_image=50)
sectioner.load_dino_model(
    arch=arch, patch_size=patch_size, pretrained_weights=dino_model_path
)
sectioner.build_sample_texton_set(threshold=threshold)
sectioner.train_clusterers()
out_dir = (
    data_root_directory().parent
    / "sectioning_out"
    / f'{"_".join(filter_type)}_{arch}_{patch_size}_{total_samples}'
)
out_dir.mkdir(parents=True, exist_ok=True)
sectioner.save(out_dir / "model.pkl")

f, axes = plt.subplots(1, 3, figsize=[16, 10])
axes = np.ravel(axes)

im = sectioner.dataset.reset_indices()
im = sectioner.dataset.stream_next()
plt.figure()
plt.imshow(im)
plt.title("Original slice")
dino_attn = sectioner.single_image_texton_descriptors(im, threshold=500)
for attn, ax in zip(dino_attn, axes):
    ax.imshow(attn)
single_meanshift_im = sectioner.label_im(im, threshold=500)
plt.figure()
plt.imshow(single_meanshift_im, cmap=plt.jet())
plt.title("Single MeanShift labels")
hierarchical_meanshift_im = sectioner.label_im(
    im, threshold=500, full_sub_structure=True
)
plt.figure()
plt.imshow(hierarchical_meanshift_im, cmap="prism", interpolation="nearest")
plt.title("Hierarchical MeanShift labels")
