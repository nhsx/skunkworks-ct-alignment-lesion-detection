"""
Tries finding lesions by doing ellipse detection on the sectioned scans.
"""
from ai_ct_scans import sectioning
import matplotlib.pyplot as plt
import numpy as np
from ai_ct_scans import data_loading
from ai_ct_scans import phase_correlation
from ai_ct_scans.data_loading import data_root_directory
from scipy.signal import medfilt2d

plt.ion()

threshold = 500

### setting up sectioner
filter_type = ["intensity"]
total_samples = 5000
model_path = (
    data_root_directory().parent
    / "sectioning_out"
    / f'{"_".join(filter_type)}_{total_samples}'
    / "model.pkl"
)
sectioner = sectioning.TextonSectioner()
sectioner.load(model_path)

ellipsoid_fitter = sectioning.CTEllipsoidFitter(
    min_area_ratio=0.75,
    min_ellipse_long_axis=10,
    max_ellipse_long_axis=200,
    max_area=25000,
)

sectioner_kwargs = {"full_sub_structure": True, "threshold": threshold}
# applying a medfilt2d can round the edges of sectioned tissues so that they get detected
# as ellipsoids a bit more easily
filterer = medfilt2d
filter_kernel = (3, 3)
single_patient = 14
patient_nums = [i for i in range(single_patient, single_patient + 1)]
for patient_num in patient_nums:
    # for body_part in ['abdo', 'thorax']:
    for body_part in ["abdo"]:
        patient_dir = data_loading.data_root_directory() / f"{patient_num}"

        patient_loader = data_loading.PatientLoader(patient_dir, rescale_on_load=False)

        if body_part == "abdo":
            scans = [patient_loader.abdo.scan_1, patient_loader.abdo.scan_2]
        else:
            scans = [patient_loader.thorax.scan_1, patient_loader.thorax.scan_2]

        for scan in scans:
            scan.load_scan()

        if patient_num == 2:
            # set up a hand-picked position for phase correlation alignment on patient 2
            orig_slice = 222
            orig_slice = int(
                orig_slice
                * scans[0].mean_z_thickness
                / scans[0].rescaled_layer_thickness
            )
            local_coords = [orig_slice, 275, 225]
        else:
            local_coords = None
        full_views = [scan.full_scan for scan in scans]
        for i, _ in enumerate(full_views):
            full_views[i][full_views[i] < threshold] = 0

        if local_coords is not None:
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
        sectioned_views = []
        ellipsoid_volumes = []
        ellipsoid_lists = []
        for view_i, full_view in enumerate(full_views):
            (
                ellipsoid_volume,
                ellipsoid_list,
                sectioned,
            ) = ellipsoid_fitter.draw_ellipsoid_walls(
                full_view,
                sectioner=sectioner,
                sectioner_kwargs=sectioner_kwargs,
                filterer=filterer,
                filter_kernel=filter_kernel,
                return_sectioned=True,
            )
            ellipsoid_volumes.append(ellipsoid_volume)
            sectioned_views.append(sectioned)
            ellipsoid_lists.append(ellipsoid_list)


def _get_centres(list_of_ellipsoids):
    centres = [element["centre"] for element in list_of_ellipsoids]
    return np.stack(centres, axis=0)


# plot the edges of ellipsoids that have been found in more than one axis and the centres in 3D
f = plt.figure()
axes = [f.add_subplot(1, 2, i + 1, projection="3d") for i in range(2)]
loc_sets = [np.where(ellipsoid > 1) for ellipsoid in ellipsoid_volumes]
centre_lists = [_get_centres(ellipsoid_list) for ellipsoid_list in ellipsoid_lists]
for ax, loc_set, centre_list in zip(axes, loc_sets, centre_lists):
    ax.scatter(loc_set[0], loc_set[1], loc_set[2], marker="o", alpha=0.004)
    ax.scatter(
        centre_list[:, 0],
        centre_list[:, 1],
        centre_list[:, 2],
        marker="x",
        color="r",
        alpha=0.5,
    )
    ax.set_xlabel("axial")
    ax.set_ylabel("coronal")
    ax.set_zlabel("sagittal")
assert True
