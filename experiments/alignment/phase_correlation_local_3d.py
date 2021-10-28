"""
Demonstrate the local phase correlation 3D method
This script cycles through sequential scans for body parts for patients, aligning on a fixed point in the scan with a
small region around that point. Points have been chosen for the first 16 patients (10 not included, which suffered from
a rescaling issue at the time of creating this script), with many values taking default values (e.g. 'centre' and 255)

"""

from pathlib import Path

from ai_ct_scans import data_loading
import matplotlib.pyplot as plt
import numpy as np
from ai_ct_scans import phase_correlation
from ai_ct_scans import phase_correlation_image_processing


plt.ion()

rois = {
    1: {"abdo": [360, 255, 155], "thorax": [200, 255, 200]},
    2: {"abdo": [255, "centre", 200], "thorax": [101, "centre", 101]},
    3: {"abdo": [550, 255, 120], "thorax": [160, 255, 225]},
    4: {"abdo": [200, 255, 220], "thorax": [300, 255, 200]},
    5: {"abdo": [500, 255, 110], "thorax": [175, 255, 210]},
    6: {"abdo": "default", "thorax": "default"},
    7: {"abdo": [420, "centre", 130], "thorax": "default"},
    8: {"abdo": [515, "centre", 140], "thorax": [150, "centre", 175]},
    9: {"abdo": [230, "centre", 220], "thorax": [200, "centre", 220]},
    11: {"abdo": [500, "centre", 120], "thorax": [150, "centre", 175]},
    13: {"abdo": [520, "centre", 120], "thorax": [150, "centre", 175]},
    14: {"abdo": [255, "centre", 170], "thorax": [150, "centre", 175]},
    15: {"abdo": [580, "centre", 170], "thorax": [250, "centre", 160]},
    16: {"abdo": [101, "centre", 200], "thorax": [250, "centre", 160]},
}

thresh = 500  # zero anything below this - a lot of noise at the 'top' of coronal views
for patient_num in range(1, 17):
    for body_part in ["abdo", "thorax"]:
        save_dir = (
            Path(__file__).parents[2]
            / "extra_data"
            / "figures"
            / "local_phase_corr_alignment_3d"
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

        f, axes = plt.subplots(1, 3, figsize=[16, 8])
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
        axes[0].set_title(f"Patient {patient_num} scan 1")
        axes[1].set_title("Scan 2")
        axes[2].set_title(
            "Scans with overlay\n Central axial plane assumed, no correction\nMost basic overlay method"
        )
        plt.tight_layout()
        curr_path = save_dir / f"central_plane.png"
        plt.savefig(curr_path)

        # roll to align central coronal plane
        shift = np.array(
            [0, central_coronal_indices[1] - central_coronal_indices[0], 0]
        )
        full_views[1] = phase_correlation.shift_nd(full_views[i], -shift)
        coronal_views = [
            scan.full_scan[:, central_coronal_indices[0], :] for scan in scans
        ]

        try:
            local_coords = rois[patient_num][body_part]
            if local_coords[1] == "centre":
                local_coords[1] = central_coronal_indices[0]
        except KeyError:
            local_coords = [255, central_coronal_indices[0], 255]
        if local_coords == "default":
            local_coords = [255, central_coronal_indices[0], 255]
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

        coronal_views_shifted = [
            np.copy(view[:, local_coords[1], :]) for view in full_views
        ]

        f_2, axes_2 = plt.subplots(1, 3, figsize=[16, 8])
        axes_2 = np.ravel(axes_2)

        for ax, scan in zip(axes_2, coronal_views_shifted):
            ax.imshow(scan)
        overlaid = phase_correlation_image_processing.generate_overlay_2d(
            coronal_views_shifted
        )
        axes_2[2].imshow(overlaid)
        axes_2[0].plot(
            local_coords[2],
            local_coords[0],
            "x",
            color="red",
            markersize=20,
            linewidth=5,
        )
        axes_2[0].set_title(f"Patient {patient_num} scan 1\nRegion of interest marked")
        axes_2[1].set_title("Scan 2")
        axes_2[2].set_title("Scans overlaid with\nfocus on small region")
        plt.tight_layout()
        curr_path = save_dir / f"central_plane_corrected.png"
        plt.savefig(curr_path)
        plt.pause(0.1)
        plt.close("all")
