"""
Tries finding lesions by doing blob detection on the raw scans. Results fairly unimpressing - finds some blobs, but they
don't seem to be detected in each consecutive axial slice very well at all, blobs will pop
in and out of existence as you scan through the stack
"""
import logging

import matplotlib.pyplot as plt
import numpy as np
from ai_ct_scans import data_loading
from ai_ct_scans import phase_correlation
import cv2
from tqdm import tqdm

plt.ion()

threshold = 500

params = cv2.SimpleBlobDetector_Params()
params.minThreshold = 5
params.maxThreshold = 255

params.filterByArea = True
params.minArea = 25
params.maxArea = 10000

params.filterByCircularity = True
params.minCircularity = 0.1

params.filterByConvexity = True
params.minConvexity = 0.5

params.filterByInertia = True
params.minInertiaRatio = 0.1

params.thresholdStep = 2.0

blob_detector = cv2.SimpleBlobDetector.create(params)
for patient_num in range(2, 3):
    # for body_part in ["abdo", "thorax"]:
    for body_part in ["abdo"]:
        patient_dir = data_loading.data_root_directory() / f"{patient_num}"

        patient_loader = data_loading.PatientLoader(patient_dir)

        if body_part == "abdo":
            scans = [patient_loader.abdo.scan_1, patient_loader.abdo.scan_2]
        else:
            # scans = [patient_loader.thorax.scan_1, patient_loader.thorax.scan_2]
            logging.error("Only one body part is supported at this time: abdo")

        for scan in scans:
            scan.load_scan()

        if patient_num == 2:
            orig_slice = 222
            local_coords = [orig_slice, 275, 225]
        full_views = [scan.full_scan for scan in scans]
        for i, _ in enumerate(full_views):
            full_views[i][full_views[i] < threshold] = 0

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
        threshold = threshold / max([full_view.max() for full_view in full_views])
        for i in range(len(full_views)):
            full_views[i] = (full_views[i] / full_views[i].max() * 255).astype("uint8")
        blob_views = [np.zeros(full_view.shape) for full_view in full_views]
        for slice_i in tqdm(range(min(full_views[0].shape[0], full_views[1].shape[0]))):
            ims = [full_view[slice_i, :, :] for full_view in full_views]
            if ims[0].max() < threshold or ims[1].max() < threshold:
                continue
            blob_ims = [np.zeros_like(im) for im in ims]
            keypoint_sets = [
                blob_detector.detect(im) + blob_detector.detect(255 - im) for im in ims
            ]

            blob_ims = [
                cv2.drawKeypoints(
                    im,
                    keypoints,
                    np.array([]),
                    (0, 0, 255),
                    cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS,
                )
                for im, keypoints in zip(blob_ims, keypoint_sets)
            ]
            for blob_view, blob_im in zip(blob_views, blob_ims):
                blob_view[slice_i, :, :] = blob_im[:, :, 2]

f = plt.figure()
axes = [f.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
loc_sets = [np.where(blob_view == 255) for blob_view in blob_views]

for ax, loc_set in zip(axes, loc_sets):
    ax.scatter(loc_set[0], loc_set[1], loc_set[2], marker="o", alpha=0.04)

f, axes = plt.subplots(1, 1)
axes = np.ravel(axes)
for i in range(orig_slice - 5, orig_slice):
    axes[0].imshow(0.25 * blob_views[0][i, :, :] + full_views[0][i, :, :])
    plt.pause(0.5)
