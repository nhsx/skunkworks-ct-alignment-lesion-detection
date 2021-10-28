"""Script to generate some example comparisons between the 2D alignment techniques."""
import os

import numpy as np
from matplotlib import pyplot as plt

from ai_ct_scans import (
    data_loading,
    keypoint_alignment,
    image_processing_utils,
    phase_correlation,
    non_rigid_alignment,
)
from ai_ct_scans.phase_correlation_image_processing import generate_overlay_2d

plt.rcParams["figure.figsize"] = [20, 10]


thresh = 500
rois = {
    1: {"abdo": [360, 255, 155], "thorax": [200, 255, 200]},
    2: {
        "abdo": [255, "centre", 200],
        "thorax": [101, "centre", 101],
    },  # patient 2 thorax is a bit awkward 2: {'abdo': [255, 255, 200], 'thorax': [250, 255, 160]},
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
    16: {"abdo": [550, "centre", 200], "thorax": [250, "centre", 160]},
}


dir_path = "all_alignment"
if not os.path.isdir(dir_path):
    os.mkdir(dir_path)


dl = data_loading.MultiPatientLoader()
path = "extra_data/data"


for i in range(3, 11):
    print(f"Patient {i}")

    patient_dir = data_loading.data_root_directory() / f"{i}"
    patient_loader = data_loading.PatientLoader(patient_dir)

    for part in ["abdo", "thorax"]:
        scans = (
            [patient_loader.abdo.scan_1, patient_loader.abdo.scan_2]
            if part == "abdo"
            else [patient_loader.thorax.scan_1, patient_loader.thorax.scan_2]
        )

        for scan in scans:
            scan.load_scan()

        # thresh out noise
        full_views = [scan.full_scan for scan in scans]
        for j, _ in enumerate(full_views):
            full_views[j][full_views[j] < thresh] = 0

        central_coronal_indices = [int(scan.full_scan.shape[1] / 2) for scan in scans]

        # roll to align central coronal plane
        shift = np.array(
            [0, central_coronal_indices[1] - central_coronal_indices[0], 0]
        )
        full_views[1] = phase_correlation.shift_nd(full_views[1], -shift)
        coronal_views = [
            scan.full_scan[:, central_coronal_indices[0], :] for scan in scans
        ]

        try:
            local_coords = rois[i][part]
            if local_coords[1] == "centre":
                local_coords[1] = central_coronal_indices[0]
        except KeyError:
            local_coords = [255, central_coronal_indices[0], 255]
        if local_coords == "default":
            local_coords = [255, central_coronal_indices[0], 255]
        region_widths = (100, 100, 100)

        reference_image = image_processing_utils.normalise(
            full_views[0][:, local_coords[1], :]
        )
        to_align = image_processing_utils.normalise(
            full_views[1][:, local_coords[1], :]
        )

        aligned_image_sift = keypoint_alignment.align_image(
            to_align, reference_image, "SIFT"
        )
        aligned_image_orb = keypoint_alignment.align_image(
            to_align, reference_image, "ORB"
        )

        shifts = phase_correlation.shifts_via_local_region(
            full_views,
            local_coords=local_coords,
            region_widths=region_widths,
            apply_lmr=True,
            apply_zero_crossings=True,
            lmr_radius=3,
        )

        for j, (shift, view) in enumerate(zip(shifts, full_views)):
            if j == 0:
                continue
            full_views[j] = phase_correlation.shift_nd(full_views[j], -shift)

        coronal_views_shifted = [
            np.copy(view[:, local_coords[1], :]) for view in full_views
        ]

        # Non-rigid alignment
        non_rigid_aligned = non_rigid_alignment.align_2D_using_CPD(
            to_align, reference_image
        )
        non_rigid_aligned = image_processing_utils.normalise(non_rigid_aligned)

        f, axarr = plt.subplots(1, 4)
        axarr[0].imshow(generate_overlay_2d([reference_image, to_align], False))
        axarr[0].title.set_text("Before alignment")
        axarr[1].imshow(
            generate_overlay_2d([reference_image, aligned_image_sift], False)
        )
        axarr[1].title.set_text("After alignment with SIFT")
        axarr[2].imshow(generate_overlay_2d(coronal_views_shifted))
        axarr[2].title.set_text("After alignment with phase correlation")
        axarr[3].imshow(
            generate_overlay_2d([reference_image, non_rigid_aligned], False)
        )
        axarr[3].title.set_text("After alignment with polynomial transform")
        plt.savefig(f"{dir_path}/aligned_{part}_{i}.png")

        for scan in scans:
            scan.full_scan = None
