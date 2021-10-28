"""
This script is used to experiment with the inputs of TextonSectioner, to find a set of filters that enable clustering
methods to label pixels on a per-pixel basis with an automatically-discovered tissue type.
Clustering on raw intensity values always showed best results, regardless of clusterer set.
Trialled clusterers included BayesianGaussianMixture, GaussianMixtures, MiniBatchKMeans,
"""

from ai_ct_scans import sectioning
import matplotlib.pyplot as plt
import numpy as np
from ai_ct_scans.data_loading import data_root_directory
from ai_ct_scans.sectioning import MeanShiftWithProbs, HierarchicalMeanShift
from ai_ct_scans import phase_correlation, phase_correlation_image_processing

plt.ion()

"""
Experimented with extracting the kernels trained at the first layer of the convolutional encoder to use as kernels for
texton descriptors. Much like Gabor and exponentially decaying cosine filters, showed reduced sectioning performance
compared to simple intensity clustering
# from ai_ct_scans.model_trainers import InfillTrainer
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from ai_ct_scans.data_loading import data_root_directory
#
# if torch.cuda.is_available():
#     dev = torch.device('cuda')
# else:
#     dev = 'cpu'
#
# save_dir = data_root_directory().parent / 'infiller_debug'
#
# trainer = InfillTrainer(num_encoder_convs=6, num_decoder_convs=6, batch_size=3, clear_previous_memmaps=False,
#                         save_dir=save_dir, save_freq=1000, encoder_filts_per_layer=24, decoder_filts_per_layer=24,
#                         num_dense_layers=1, neurons_per_dense=128, learning_rate=0.00001, blank_width=16, kernel_size=7)
# trainer.load_model(trainer.save_dir)
# ###########
# kernels = np.copy(np.squeeze(trainer.model.state_dict()['encoder_convs.1.layers.0.weight'].cpu().detach().numpy()))
# del trainer
"""

# set up the sectioner and train
filter_type = ["intensity"]
medfilt_kernel = None
blur_kernel = None
total_samples = 5000
if blur_kernel is not None:
    blur_str_list = [str(el) for el in blur_kernel]
    dir_name = "_".join(filter_type + blur_str_list)
    out_dir = (
        data_root_directory().parent / "sectioning_out" / f"{dir_name}_{total_samples}"
    )
else:
    out_dir = (
        data_root_directory().parent
        / "sectioning_out"
        / f'{"_".join(filter_type)}_{total_samples}'
    )
out_dir.mkdir(parents=True, exist_ok=True)

kernels = None
threshold = 500
np.random.seed(555)

clusterers = [HierarchicalMeanShift()]
clusterer_titles = ["HierarchicalMeanShift"]

sectioner = sectioning.TextonSectioner(
    filter_type=filter_type,
    total_samples=total_samples,
    samples_per_image=50,
    kernels=kernels,
    blur_kernel=blur_kernel,
    clusterers=clusterers,
    clusterer_titles=clusterer_titles,
    medfilt_kernel=medfilt_kernel,
)
sectioner.build_sample_texton_set(threshold=threshold)
sectioner.train_clusterers()

# reset indices inside the random sampling, otherwise it can confuse the rescaling of full data
sectioner.dataset.reset_indices()

# set some phase correlation alignment parameters depending on which patient is selected - just hand picked for this
# script
patient_ind = 0  # off by 1 between index and string label, e.g. 0 is patient 1
if patient_ind == 0:
    orig_slice = 112
    local_coords = [orig_slice, 275, 225]
elif patient_ind == 9 or patient_ind == 1:
    orig_slice = 222
    local_coords = [orig_slice, 275, 225]
elif patient_ind == 3:
    orig_slice = 310
    local_coords = [orig_slice, 300, 250]
scan_1 = sectioner.dataset.multi_patient_loader.patients[patient_ind].abdo.scan_1
scan_2 = sectioner.dataset.multi_patient_loader.patients[patient_ind].abdo.scan_2
scan_1.load_scan()
scan_2.load_scan()
full_views = [scan.full_scan for scan in [scan_1, scan_2]]
for i, _ in enumerate(full_views):
    full_views[i][full_views[i] < threshold] = 0

# align the full scans
region_widths = (100, 100, 100)
shifts = phase_correlation.shifts_via_local_region(
    full_views,
    local_coords=local_coords,
    region_widths=region_widths,
    apply_lmr=True,
    apply_zero_crossings=True,
    lmr_radius=3,
)
for i, (shift, view) in enumerate(zip(shifts, full_views)):
    if i == 0:
        continue
    full_views[i] = phase_correlation.shift_nd(full_views[i], -shift)

# extract some slices
axial_slice = int(
    orig_slice * scan_1.mean_z_thickness / scan_1.rescaled_layer_thickness
)
imp = full_views[0][axial_slice]
imp_2 = full_views[1][axial_slice]

im = scan_1.load_2d_array(orig_slice)
im_2 = scan_2.load_2d_array(orig_slice)
plt.figure()
plt.imshow(im, cmap="gray")
plt.title("orig slice scan 1")
plt.figure()
plt.imshow(im_2, cmap="gray")
plt.title("orig slice scan 2")

# Have a look at various different outputs of the sectioner - probabilities images are particularly interesting
# and didn't come up during stakeholder review much

clusterer_ind = 0
cluster_label = 1
out_im, probs = sectioner.probabilities_im(
    imp, threshold=threshold, clusterer_ind=clusterer_ind
)
out_im_2, probs_2 = sectioner.probabilities_im(
    imp_2, threshold=threshold, clusterer_ind=clusterer_ind
)
plt.figure()
plt.imshow(imp, cmap="gray")
plt.title(f"scan 1 slice {orig_slice}")
plt.figure()
plt.imshow(imp_2, cmap="gray")
plt.title(f"scan 2 slice {orig_slice}")
plt.figure()
plt.imshow(out_im, cmap=plt.jet())
plt.title(f"scan 1 slice {orig_slice} classes")
plt.figure()
plt.imshow(out_im_2, cmap=plt.jet())
plt.title(f"scan 2 slice {orig_slice} classes")
plt.figure()
plt.imshow(probs[cluster_label] * (out_im == cluster_label), cmap="gray")
plt.title(f"scan 1 slice {orig_slice} probabilities of class {cluster_label}")
plt.figure()
plt.imshow(probs_2[cluster_label] * (out_im_2 == cluster_label), cmap="gray")
plt.title(f"scan 2 slice {orig_slice} probabilities of class {cluster_label}")

sub_classes_1 = sectioner.label_im(
    imp,
    threshold=threshold,
    clusterer_ind=clusterer_ind,
    sub_structure_class_label=cluster_label,
)
plt.figure()
plt.imshow(sub_classes_1, interpolation="nearest")
plt.title(f"Subclasses scan 1 cluster {cluster_label}")
sub_classes_2 = sectioner.label_im(
    imp_2,
    threshold=threshold,
    clusterer_ind=clusterer_ind,
    sub_structure_class_label=cluster_label,
)
plt.figure()
plt.imshow(sub_classes_2, interpolation="nearest")
plt.title(f"Subclasses scan 2 cluster {cluster_label}")

sub_cluster_label = 1
_, sub_classes_1_prob = sectioner.probabilities_im(
    imp,
    threshold=threshold,
    clusterer_ind=clusterer_ind,
    cluster_label=cluster_label,
    return_sub_structure=True,
    sub_structure_class_label=sub_cluster_label,
)
sub_class_1_prob_sectioned = sub_classes_1_prob * (sub_classes_1 == sub_cluster_label)
plt.figure()
plt.imshow(sub_class_1_prob_sectioned, cmap="gray", interpolation="nearest")
plt.title(
    f"Subclasses probabilities scan 1 cluster {cluster_label} subcluster {sub_cluster_label}"
)
_, sub_classes_2_prob = sectioner.probabilities_im(
    imp_2,
    threshold=threshold,
    clusterer_ind=clusterer_ind,
    cluster_label=cluster_label,
    return_sub_structure=True,
    sub_structure_class_label=sub_cluster_label,
)
sub_class_2_prob_sectioned = sub_classes_2_prob * (sub_classes_2 == sub_cluster_label)
plt.figure()
plt.imshow(sub_class_2_prob_sectioned, cmap="gray", interpolation="nearest")
plt.title(
    f"Subclasses probabilities scan 2 cluster {cluster_label} subcluster {sub_cluster_label}"
)

overlay = phase_correlation_image_processing.generate_overlay_2d(
    [sub_class_1_prob_sectioned, sub_class_2_prob_sectioned], normalize=False
)
overlay = overlay / overlay.max()
plt.figure()
plt.imshow(overlay)

full_sub_classes_1 = sectioner.label_im(
    imp, threshold=threshold, clusterer_ind=0, full_sub_structure=True
)
plt.figure()
plt.imshow(full_sub_classes_1, cmap="hsv", interpolation="nearest")
plt.title(f"Full subclasses scan 1")
full_sub_classes_2 = sectioner.label_im(
    imp_2, threshold=threshold, clusterer_ind=0, full_sub_structure=True
)
plt.figure()
plt.imshow(full_sub_classes_2, cmap="hsv", interpolation="nearest")
plt.title(f"Full subclasses scan 2")

# MeanShiftWithProbs and HierarchicalMeanShift have standard deviations for each class that can be used to
# automatically set colour map thresholds, much like 'windows' in existing CT analysis software but
# automatically discovered in the dataset
if isinstance(sectioner.clusterers[clusterer_ind], MeanShiftWithProbs):
    vmin = (
        sectioner.clusterers[clusterer_ind].cluster_centers_[cluster_label]
        - 2 * sectioner.clusterers[clusterer_ind].deviations[cluster_label]
    )
    vmax = (
        sectioner.clusterers[clusterer_ind].cluster_centers_[cluster_label]
        + 2 * sectioner.clusterers[clusterer_ind].deviations[cluster_label]
    )
elif isinstance(sectioner.clusterers[clusterer_ind], HierarchicalMeanShift):
    vmin = (
        sectioner.clusterers[clusterer_ind].base_clusterer.cluster_centers_[
            cluster_label
        ]
        - 2
        * sectioner.clusterers[clusterer_ind].base_clusterer.deviations[cluster_label]
    )
    vmax = (
        sectioner.clusterers[clusterer_ind].base_clusterer.cluster_centers_[
            cluster_label
        ]
        + 2
        * sectioner.clusterers[clusterer_ind].base_clusterer.deviations[cluster_label]
    )
else:
    # just set some defaults if not using MeanShiftWithProbs or HierarchicalMeanShift, expecting these to be changed
    # with which class you want to look at
    vmin = 1000
    vmax = 1300

plt.figure()
plt.imshow(im, cmap="gray", vmin=vmin, vmax=vmax)
plt.title(f"scan 1 slice {orig_slice} unaligned classes tuned colour range")
plt.figure()
plt.imshow(im_2, cmap="gray", vmin=vmin, vmax=vmax)
plt.title(f"scan 2 slice {orig_slice} unaligned classes tuned colour range")

overlay = phase_correlation_image_processing.generate_overlay_2d(
    [out_im, out_im_2], normalize=False
)
overlay = overlay / overlay.max()
plt.figure()
plt.imshow(overlay)
overlay = phase_correlation_image_processing.generate_overlay_2d(
    [imp, imp_2], normalize=False
)
overlay = overlay / overlay.max()
plt.figure()
plt.imshow(overlay)
plt.title("Overlay of original images aligned")

### 1 - probability to make improbable tissues bright
inv_prob = (1 - probs[cluster_label]) * (out_im == cluster_label)
inv_prob[imp < threshold] = 0
inv_prob_2 = (1 - probs_2[cluster_label]) * (out_im_2 == cluster_label)
inv_prob_2[imp_2 < threshold] = 0
overlay = phase_correlation_image_processing.generate_overlay_2d(
    [inv_prob, inv_prob_2], normalize=False
)
overlay = overlay / overlay.max()
plt.figure()
plt.imshow(overlay)
plt.figure()
plt.imshow(inv_prob, cmap="gray")
plt.title("1 - prob scan 1")
plt.figure()
plt.imshow(inv_prob_2, cmap="gray")
plt.title("1 - prob scan 2")

assert True
