"""
A script to trial phase correlation for aligning 2D slices, runs through the first 5 patients and saves images for
each body part on each of them. This is not expected to generate results as impressive as the local 3D phase
correlation method, but demonstrates the method can work in 2D
"""

from pathlib import Path

from ai_ct_scans import data_loading
import matplotlib.pyplot as plt
import numpy as np
from ai_ct_scans import phase_correlation
from ai_ct_scans import phase_correlation_image_processing

plt.ion()

thresh = 500
for patient_num in range(1, 6):
    for body_part in ["abdo", "thorax"]:
        save_dir = (
            Path(__file__).parents[2]
            / "extra_data"
            / "figures"
            / "global_phase_corr_alignment_2d"
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

        f, axes = plt.subplots(1, 3, figsize=[16, 8])
        axes = np.ravel(axes)

        scans[0].load_scan()
        scans[1].load_scan()

        # thresh out noise
        for i, _ in enumerate(scans):
            scans[i].full_scan[scans[i].full_scan < thresh] = 0

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

        shift = phase_correlation.shift_via_phase_correlation_nd(
            coronal_views, lmr_radius=3, apply_zero_crossings=False
        )

        coronal_views[1] = phase_correlation.shift_nd(
            coronal_views[1], -np.array(shift[1])
        )

        f_2, axes_2 = plt.subplots(1, 3, figsize=[16, 8])
        axes_2 = np.ravel(axes_2)

        for ax, scan in zip(axes_2, coronal_views):
            ax.imshow(scan)
        overlaid = phase_correlation_image_processing.generate_overlay_2d(coronal_views)
        axes_2[2].imshow(overlaid)
        axes_2[0].set_title("Scan 1")
        axes_2[1].set_title("Scan 2")
        axes_2[2].set_title(
            "Scans with overlay\nFull 2D view phase correlation alignment correction\nAlignment of major structures dominates,\nlocal differences possible due to body shape changes"
        )
        plt.tight_layout()
        curr_path = save_dir / f"central_plane_phase_corr.png"
        plt.savefig(curr_path)
        plt.pause(0.1)
        plt.close("all")
