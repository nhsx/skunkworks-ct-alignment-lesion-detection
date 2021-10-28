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

patient_num = str(2)
scan_dir = data_loading.data_root_directory() / f"{patient_num}/{patient_num}/Abdo1"
anomaly_path = (
    data_loading.data_root_directory().parent
    / "infiller_with_blur_64_mask"
    / f"patient_{patient_num}_anomaly_test__606_396_396.npy"
)

scan_loader = data_loading.ScanLoader(scan_dir)
scan_loader.load_scan()
scan = scan_loader.full_scan
anomaly_scan = data_loading.load_memmap(anomaly_path)

norm_anom_scan = anomaly_scan / (scan.mean() + np.abs(scan - scan.mean()))
abs_anom_scan = np.abs(anomaly_scan) * (scan > 500)
threshed_anom_scan = norm_anom_scan * (scan > 500)

f, axes = plt.subplots(1, 5)
axes = np.ravel(axes)
for ax, im in zip(
    axes, [scan, anomaly_scan, norm_anom_scan, threshed_anom_scan, abs_anom_scan]
):
    ax.imshow(im[222, :, :])

assert True
