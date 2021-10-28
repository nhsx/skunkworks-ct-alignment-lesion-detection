"""Script to test 3d alignment techniques on a batch of scans."""
import os

from matplotlib import pyplot as plt

from ai_ct_scans import data_loading, non_rigid_alignment
from ai_ct_scans.phase_correlation_image_processing import generate_overlay_2d
from ai_ct_scans.image_processing_utils import normalise

plt.rcParams["figure.figsize"] = [20, 10]


trans_dir_path = "alignment_transforms"
if not os.path.isdir(trans_dir_path):
    os.mkdir(trans_dir_path)

image_dir_path = "alignment_images"
if not os.path.isdir(image_dir_path):
    os.mkdir(image_dir_path)


dl = data_loading.MultiPatientLoader()
path = "extra_data/data"


for i in range(1, 21):
    patient_dir = data_loading.data_root_directory() / f"{i}"
    patient_loader = data_loading.PatientLoader(patient_dir)
    patient_loader.abdo.scan_1.load_scan()
    patient_loader.abdo.scan_2.load_scan()

    trans = non_rigid_alignment.estimate_3D_alignment_transform(
        patient_loader.abdo.scan_2.full_scan,
        patient_loader.abdo.scan_1.full_scan,
        maximum_source_points=5000,
        maximum_target_points=50000,
    )
    non_rigid_alignment.write_transform(
        trans, os.path.join(trans_dir_path, f"patient_{i}.pkl")
    )
    aligned = non_rigid_alignment.transform_3d_volume(
        patient_loader.abdo.scan_2.full_scan, trans.predict
    )

    # Plot a set of examples
    image_dir_path = os.path.join("alignment_images", f"patient_{i}")
    if not os.path.isdir(image_dir_path):
        os.mkdir(image_dir_path)

    try:
        for slice in [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]:
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(
                generate_overlay_2d(
                    [
                        normalise(patient_loader.abdo.scan_1.full_scan[slice, :, :]),
                        normalise(patient_loader.abdo.scan_2.full_scan[slice, :, :]),
                    ],
                    False,
                )
            )
            axarr[0].title.set_text("Before alignment")
            axarr[1].imshow(
                generate_overlay_2d(
                    [
                        normalise(patient_loader.abdo.scan_1.full_scan[slice, :, :]),
                        normalise(aligned[slice, :, :]),
                    ],
                    False,
                )
            )
            axarr[1].title.set_text("After non-rigid alignment")
            f.savefig(os.path.join(image_dir_path, f"axial_slice_{slice}.png"))

        for slice in [50, 100, 150, 200, 250, 300, 350, 400]:
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(
                generate_overlay_2d(
                    [
                        normalise(patient_loader.abdo.scan_1.full_scan[:, slice, :]),
                        normalise(patient_loader.abdo.scan_2.full_scan[:, slice, :]),
                    ],
                    False,
                )
            )
            axarr[0].title.set_text("Before alignment")
            axarr[1].imshow(
                generate_overlay_2d(
                    [
                        normalise(patient_loader.abdo.scan_1.full_scan[:, slice, :]),
                        normalise(aligned[:, slice, :]),
                    ],
                    False,
                )
            )
            axarr[1].title.set_text("After non-rigid alignment")
            f.savefig(os.path.join(image_dir_path, f"coronal_slice_{slice}.png"))

        for slice in [50, 100, 150, 200, 250, 300, 350, 400]:
            f, axarr = plt.subplots(1, 2)
            axarr[0].imshow(
                generate_overlay_2d(
                    [
                        normalise(patient_loader.abdo.scan_1.full_scan[:, :, slice]),
                        normalise(patient_loader.abdo.scan_2.full_scan[:, :, slice]),
                    ],
                    False,
                )
            )
            axarr[0].title.set_text("Before alignment")
            axarr[1].imshow(
                generate_overlay_2d(
                    [
                        normalise(patient_loader.abdo.scan_1.full_scan[:, :, slice]),
                        normalise(aligned[:, :, slice]),
                    ],
                    False,
                )
            )
            axarr[1].title.set_text("After non-rigid alignment")
            f.savefig(os.path.join(image_dir_path, f"sagittal_slice_{slice}.png"))
    except:
        pass
