"""
Train a tissue sectioner and save it, using the best options discovered during the project
"""

from ai_ct_scans import sectioning
import numpy as np
from ai_ct_scans.sectioning import HierarchicalMeanShift
from ai_ct_scans.data_loading import data_root_directory

filter_type = ["intensity"]
medfilt_kernel = None
blur_kernel = None
total_samples = 5000

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
# Assume only one intensity is generated/tested
# out_dir = (
#     data_root_directory().parent
#     / "sectioning_out"
#     / f'{"_".join(filter_type)}_{total_samples}'
# )
# out_dir.mkdir(parents=True, exist_ok=True)
# sectioner.save(out_dir / "model.pkl")
sectioner.save(
    data_root_directory().parent / "hierarchical_mean_shift_tissue_sectioner_model.pkl"
)
