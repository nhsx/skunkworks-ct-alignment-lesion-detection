"""
Loads 3D data that has been stitched together from predicted regions of output of an infiller model, and displays a
2D slice after some methods of normalisation.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from ai_ct_scans import data_loading

plt.ion()

if torch.cuda.is_available():
    dev = torch.device("cuda")
else:
    dev = "cpu"

scan_dir = data_loading.data_root_directory() / "1/1/Abdo1"
anomaly_path = (
    data_loading.data_root_directory().parent
    / "infiller_with_blur"
    / "patient_1_anomaly_full__672_442_442.npy"
)

scan_loader = data_loading.ScanLoader(scan_dir)
scan_loader.load_scan()
scan = scan_loader.full_scan
anomaly_scan = data_loading.load_memmap(anomaly_path)

norm_anom_scan = anomaly_scan / (scan.mean() + np.abs(scan - scan.mean()))

f, axes = plt.subplots(1, 3)
axes = np.ravel(axes)
for ax, im in zip(axes, [scan, anomaly_scan, norm_anom_scan]):
    ax.imshow(im[210, :, :])

assert True
