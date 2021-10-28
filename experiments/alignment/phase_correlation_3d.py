"""
A script to trial phase correlation for aligning full 3D scans, runs through the first 5 patients and saves images for
each body part on each of them. This is not expected to generate results as impressive as the local 3D phase
correlation method, but demonstrates the method can work on full 3D data
"""
from pathlib import Path

from ai_ct_scans import data_loading
import matplotlib.pyplot as plt
import numpy as np
from ai_ct_scans import phase_correlation
from ai_ct_scans import phase_correlation_image_processing

"""
Doesn't work very well yet - likely rescaling needs doing before it'll work well. Also slow, might do with rescaling
before shift detection for speed
"""

plt.ion()

thresh = 500  # zero anything below this - a lot of noise at the 'top' of coronal views
for patient_num in range(1, 6):
    for body_part in ["abdo", "thorax"]:
        save_dir = (
            Path(__file__).parents[2]
            / "extra_data"
            / "figures"
            / "global_phase_corr_alignment_3d"
            / f"{patient_num}"
            / f"{body_part}"
        )
        save_dir.mkdir(exist_ok=True, parents=True)
        patient_dir = data_loading.data_root_directory() / f"{patient_num}"

        patient_loader = data_loading.PatientLoader(patient_dir)

        if body_part == "abdo":
            scans = [patient_loader.abdo.scan_1, patient_loader.abdo.scan_2]
        else:
            scans = [patient_loader.thorax.scan_1, patient_loader.thorax.scan_2]

        for scan in scans:
            scan.load_scan()

        # thresh out noise
        full_views = [scan.full_scan for scan in scans]
        for i, _ in enumerate(full_views):
            full_views[i][full_views[i] < thresh] = 0

        f, axes = plt.subplots(1, 3, figsize=[16, 10])
        axes = np.ravel(axes)

        central_coronal_indices = [int(scan.full_scan.shape[1] / 2) for scan in scans]
        coronal_views = [
            scan.full_scan[:, central_coronal_index, :]
            for scan, central_coronal_index in zip(scans, central_coronal_indices)
        ]

        for ax, view in zip(axes, coronal_views):
            ax.imshow(view)
        overlaid = phase_correlation_image_processing.generate_overlay_2d(coronal_views)
        axes[2].imshow(overlaid)
        axes[0].set_title("Scan 1")
        axes[1].set_title("Scan 2")
        axes[2].set_title(
            "Scans with overlay\n Central axial plane assumed, no correction\nMost basic overlay method"
        )
        plt.tight_layout()
        curr_path = save_dir / f"central_plane.png"
        plt.savefig(curr_path)

        shifts = phase_correlation.shift_via_phase_correlation_nd(
            full_views,
            apply_zero_crossings=False,
            lmr_radius=3,
        )

        for i, (shift, view) in enumerate(zip(shifts, full_views)):
            if i == 0:
                continue
            full_views[i] = phase_correlation.shift_nd(full_views[i], -shift)

        coronal_views = [
            view[:, int(full_views[0].shape[1] / 2), :] for view in full_views
        ]

        f_2, axes_2 = plt.subplots(1, 3, figsize=[16, 10])
        axes_2 = np.ravel(axes_2)

        for ax, view in zip(axes_2, coronal_views):
            ax.imshow(view)
        overlaid = phase_correlation_image_processing.generate_overlay_2d(coronal_views)
        axes_2[2].imshow(overlaid)
        axes_2[0].set_title("Scan 1")
        axes_2[1].set_title("Scan 2")
        axes_2[2].set_title(
            "Scans with overlay\nFull 3D view phase\ncorrelation alignment correction"
        )
        plt.tight_layout()
        curr_path = save_dir / f"central_plane_phase_corr.png"
        plt.savefig(curr_path)
        plt.pause(0.1)
        plt.close("all")
