"""
Run through the first 20 patients and check their z scale, flagging any where the z thickness in the first scan
is different to the second scan. This helped debug the axial rescaling problem that was encountered early in the
project
"""

from pathlib import Path

from ai_ct_scans import data_loading
import matplotlib.pyplot as plt


plt.ion()

thresh = 500  # zero anything below this - a lot of noise at the 'top' of coronal views
for patient_num in range(1, 21):
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
        thickness_diff = scans[0].mean_z_thickness - scans[1].mean_z_thickness
        if abs(thickness_diff) < 1e-10:
            # ignore machine precision error
            thickness_diff = 0
        print(
            f"{thickness_diff} difference between z thicknesses for patient {patient_num}, body part {body_part}"
        )
