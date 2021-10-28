"""
This script will generate a minimal set of dicom files in a folder structure similar to that of the real data,
suitable for testing.
Following the guide to DICOM file creation at
https://pydicom.github.io/pydicom/stable/auto_examples/input_output/plot_write_dicom.html
with some additional attributes set to allow for pixel array being set
"""

from pathlib import Path
import os

import pydicom
import numpy as np
from ai_ct_scans.data_writing import create_dicom_file

# test_dir = Path(__file__).parents[2] / "tests" / "fixtures" / "dicom_data"
test_dir = f"../../tests/fixtures/dicom_data"

# patient_dirs = [test_dir / f"{i}" / f"{i}" for i in range(1, 3)]
patient_dirs = [f"{test_dir}/{i}/{i}" for i in range(1, 3)]

# Focusing only one body part in this version
# body_parts = ["Abdo", "Thorax"]
body_parts = ["Abdo"]

for patient_dir in patient_dirs:
    for body_part in body_parts:
        for i in range(1, 3):
            # curr_dir = patient_dir / f"{body_part}{i}" / "DICOM"
            # curr_dir.mkdir(exist_ok=True, parents=True)
            curr_dir = f"{patient_dir}/{body_part}{i}/DICOM"
            if not os.path.exists(curr_dir):
                os.makedirs(curr_dir)
            for j in range(2):
                # curr_path = str(curr_dir / f"I{j}")
                curr_path = str(f"{curr_dir}/I{j}")
                rand_array = (np.random.rand(384, 512) * 255).astype("uint16")
                # expected slice thickness is 0.7, approx starting location of 1000. is observed often
                ds = create_dicom_file(
                    curr_path, rand_array, slice_location=(1000.0 + j * 0.7)
                )
                reloaded = pydicom.read_file(curr_path)
                np.testing.assert_array_equal(reloaded.pixel_array, rand_array)
