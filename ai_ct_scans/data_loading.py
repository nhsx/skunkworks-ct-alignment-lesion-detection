from pathlib import Path
import numpy as np
from pydicom import dcmread
import cv2
import os
from pandas import read_csv


def dir_dicom_paths(directory):
    """Return list of paths to files within a directory at any level below, beginning with 'I'. This is the naming
    convention expected for the project.

    Args:
        directory (pathlib Path): A path

    Returns:
        (list of pathlib Paths) All paths to files beginning with 'I' within the given directory

    """
    return list(directory.glob("**/I*"))


def data_root_directory():
    """Get a path to the top level data directory. Checks whether you have created the local store of data at the expected
    location and returns that as default if found, otherwise returns the location on the shared folder

    Returns:
        (pathlib Path): A path

    """
    return Path(__file__).parents[1] / "extra_data" / "data"


def _extract_memmap_shape(path):
    """Get the shape tuple from a memmap path as saved in the expected format
    [prefix]__[shape element 1]_[shape_element 2]_..._[shape element n].npy

    Args:
        path (pathlib Path): Path to the memmap

    Returns:
        (tuple of ints): The shape of the memmap

    """
    return tuple(int(element) for element in path.stem.split("__")[1].split("_"))


def load_memmap(path, dtype="float64"):
    """Load a memmap array with an expected naming convention of the form
    [prefix]__[shape element 1]_[shape_element 2]_..._[shape element n].npy

    Args:
        path (pathlib Path): path to the memmap
        dtype (str or valid numpy dtype): the data type the memmap was saved as

    Returns:
        (np.memmap): A memmap ndarray

    """
    memmap_shape = _extract_memmap_shape(path)
    return np.memmap(path, dtype=dtype, mode="r", shape=memmap_shape)


class ScanLoader:
    """A class providing intuitive loading of DICOM data, for a single scan for a single body part

    Attributes:
        dicom_paths (list of pathlib Paths): All the paths to dicom files within a given scan directory
        full_scan (np.ndarray or None): The entire scan loaded as a 3D ndarray

    """

    def __init__(self, root_dir, rescale_on_load=True):
        """
        Find the paths to DICOM data for the scan

        Args:
            root_dir (pathlib Path): A directory
            rescale_on_load (bool): Whether to rescale axial planes such that each pixel is laterally 1mm apart, and
            rescale the separation between axial slices to the standard 0.7mm spacing
        """
        if not root_dir.exists():
            raise FileNotFoundError(f"specified directory {root_dir} not found")
        dicom_paths = dir_dicom_paths(root_dir)
        disordered_stems = [int(path.stem[1:]) for path in dicom_paths]
        sorted_path_order = np.argsort(disordered_stems)
        self.dicom_paths = [dicom_paths[i] for i in sorted_path_order]
        self.full_scan = None
        self.pixel_spacing = [0.0, 0.0]
        self.rescale_on_load = rescale_on_load
        self.rescaled_transverse_shape = None
        self.memmap_save_paths = []
        self.full_memmap = None
        self.transpose_memmap = None
        self.rescaled_layer_thickness = 0.7
        self.rescaled_z = None
        self.z_locations = []
        self.mean_z_thickness = None
        self.patient_name = None

    def load_2d_array(self, index, ignore_rescale=False, ignore_z_rescale=False):
        """Loads the pixel array from the dicom at position index in self.dicom_paths

        Args:
            ignore_z_rescale (bool): If True, do not store the z location of the slice for later axial rescaling in the
            full 3D view.
            ignore_rescale (bool): If True, load raw data without rescaling, otherwise rescale in accordance with state
            of self.rescale_on_load
            index (int): Scans are ordered I0 ... IN in the original directory. This arg sets the N to load

        Returns:
            (np.ndarray) A 2D slice of the scan

        """
        if self.rescale_on_load and not ignore_rescale:
            layer = dcmread(self.dicom_paths[index])
            if not ignore_z_rescale:
                self.z_locations.append(float(layer.SliceLocation))
            return cv2.resize(
                layer.pixel_array,
                self.rescaled_transverse_shape[::-1],
                interpolation=cv2.INTER_CUBIC,
            )
        return dcmread(self.dicom_paths[index]).pixel_array

    def raw_transverse_pixel_spacing_and_shape(self):
        """Get the original pixel array shape for the transverse plane of the scan, its pixel spacing

        Returns (tuple of tuple of ints and ndarray):
            The original pixel array shape for the transverse plane of the scan and its pixel spacing

        """
        layer = dcmread(self.dicom_paths[0])
        return layer.pixel_array.shape, np.array(layer.PixelSpacing)

    def rescale_depth(self):
        """
        Rescale the full 3D scan along depth based on the ContentTime difference between last and first scans and total
        slice number

        """

        arr = np.zeros(
            [self.rescaled_z, *self.full_scan.shape[1:]], dtype=self.full_scan.dtype
        )

        for i in range(self.full_scan.shape[1]):
            arr[:, i, :] = cv2.resize(
                self.full_scan[:, i, :],
                (
                    self.full_scan.shape[2],
                    self.rescaled_z,
                ),
                interpolation=cv2.INTER_CUBIC,
            )
        self.full_scan = arr

    def mean_axial_thickness(self):
        """Get the average axial layer thickness of the scan

        Returns:
            (float): The average thickness of axial layers in the scan

        """
        z_loc_diff = np.diff(self.z_locations)
        return np.mean(z_loc_diff)

    def load_patient_metadata(self):
        """Load relevant patient metadata."""
        layer = dcmread(self.dicom_paths[0])
        self.patient_name = "{}, {}".format(
            layer.PatientName.family_name, layer.PatientName.given_name
        )

    def load_scan(self):
        """Loads the entire 3D scan into memory, optionally rescaling transverse plane views to build the stack
        according to whether self.rescale_on_load is True or False

        Returns:
            (np.ndarray): The full 3D data of the scan, excluding metadata

        """
        self.load_patient_metadata()
        if self.full_scan is None:
            (
                self.rescaled_transverse_shape,
                self.pixel_spacing,
            ) = self.raw_transverse_pixel_spacing_and_shape()

            if self.rescale_on_load:
                self.rescaled_transverse_shape = tuple(
                    np.round(
                        np.array(self.rescaled_transverse_shape) * self.pixel_spacing
                    ).astype(int)
                )
            self.full_scan = np.zeros(
                [len(self.dicom_paths), *self.rescaled_transverse_shape]
            )
        for i, path in enumerate(self.dicom_paths):
            self.full_scan[i] = self.load_2d_array(i)
        if self.rescale_on_load:

            self.mean_z_thickness = self.mean_axial_thickness()
            if abs(abs(self.mean_z_thickness) - self.rescaled_layer_thickness) > 1e-6:
                self.rescaled_z = int(
                    np.round(
                        self.full_scan.shape[0]
                        * abs(self.mean_z_thickness)
                        / self.rescaled_layer_thickness
                    )
                )

                self.rescale_depth()

    def full_scan_to_memmap(self, directory=None, normalise=True):
        """Send the 3D data to two numpy memmaps, with filename-embedded shapes that allows for reloading. Default to
        the same directory as the DICOM data. Second memmap is first and second axes (0-indexed) swapped, to enable
        faster slicing during DL training

        Args:
            normalise (bool): whether to normalise the data upon saving to memmap
            directory (pathlib Path): directory in which to save the memmap

        Returns:

        """
        if directory is None:
            directory = self.dicom_paths[0].parent
        shape_str = "_".join([str(element) for element in list(self.full_scan.shape)])
        self.memmap_save_paths.append(directory / f"orig__{shape_str}.npy")
        arr = np.memmap(
            self.memmap_save_paths[0],
            dtype=self.full_scan.dtype,
            mode="w+",
            shape=self.full_scan.shape,
        )
        if normalise is True:
            arr[:] = self.full_scan[:] / (
                max(self.full_scan.max(), 1.0) - self.full_scan.min()
            )
        else:
            arr[:] = self.full_scan[:]
        del arr

        transpose_scan = np.transpose(self.full_scan, (0, 2, 1)).copy()
        shape_str = "_".join([str(element) for element in list(transpose_scan.shape)])
        self.memmap_save_paths.append(directory / f"transpose__{shape_str}.npy")
        arr = np.memmap(
            self.memmap_save_paths[1],
            dtype=transpose_scan.dtype,
            mode="w+",
            shape=transpose_scan.shape,
        )
        if normalise is True:
            arr[:] = transpose_scan[:] / (
                max(transpose_scan.max(), 1.0) - transpose_scan.min()
            )
        else:
            arr[:] = transpose_scan[:]
        del arr

    @staticmethod
    def _npy_list(path):
        return list(path.glob("*.npy"))

    def _find_memmaps(self):
        paths = self._npy_list(self.dicom_paths[0].parent)
        if len(paths) == 0:
            return []
        else:
            return paths

    def delete_memmap(self):
        paths = self._npy_list(self.dicom_paths[0].parent)
        if paths is not None:
            for path in paths:
                os.remove(path)

    def _get_memmap_shape(self):
        return _extract_memmap_shape(self.memmap_save_paths[0])

    def load_full_memmap(self, normalise=True):
        """Loads the numpy memmapped version of the full scan, or creates it if it doesn't yet exist

        Returns (None):

        """
        if self.memmap_save_paths is None or len(self.memmap_save_paths) == 0:
            self.memmap_save_paths = self._find_memmaps()
            if len(self.memmap_save_paths) == 0:
                self.load_scan()
                self.full_scan_to_memmap(normalise=normalise)
        memmap_shape = self._get_memmap_shape()
        self.full_memmap = np.memmap(
            self.memmap_save_paths[0], dtype="float64", mode="r", shape=memmap_shape
        )

        memmap_shape = list(memmap_shape)
        memmap_shape[1], memmap_shape[2] = memmap_shape[2], memmap_shape[1]
        memmap_shape = tuple(memmap_shape)

        self.transpose_memmap = np.memmap(
            self.memmap_save_paths[1], dtype="float64", mode="r", shape=memmap_shape
        )

    def clear_scan(self):
        """Sets the full_scan to none"""
        self.full_scan = None

    def load_memmap_and_clear_scan(self, normalise=True):
        """Loads the memmap of the scan and clears the in-memory scan

        Returns (None):

        """
        self.load_full_memmap(normalise=normalise)
        self.clear_scan()


class BodyPartLoader:
    """A class providing intuitive loading of DICOM data, scan-order-wise for a particular body part

    Attributes:
        scan_1 (ScanLoader): A ScanLoader for the former scan of a particular body part
        scan_2 (ScanLoader): A ScanLoader for the latter scan of a particular body part
    """

    def __init__(self, root_dir, body_part, rescale_on_load=True):
        """
        Find the paths to DICOM data for the body part, separating by sequential scans

        Args:
            root_dir (pathlib Path): A directory
            rescale_on_load (bool): Whether to rescale axial planes such that each pixel is laterally 1mm apart, and
            rescale the separation between axial slices to the standard 0.7mm spacing
        """
        root_dirs = [root_dir / root_dir.stem / f"{body_part}{i}" for i in range(1, 3)]
        self.scan_1 = ScanLoader(root_dirs[0], rescale_on_load=rescale_on_load)
        self.scan_2 = ScanLoader(root_dirs[1], rescale_on_load=rescale_on_load)


class PatientLoader:
    """A class providing intuitive loading of DICOM data patient-wise, scan-order-wise and body-region-wise

    Attributes:
        abdo (BodyPartLoader): A BodyPartLoader for the abdomen of a patient
    """

    def __init__(self, root_dir, rescale_on_load=True):
        """
        Find the paths to DICOM data for the patient, separating by abdo and thorax

        Args:
            root_dir (pathlib Path): A directory
            rescale_on_load (bool): Whether to rescale axial planes such that each pixel is laterally 1mm apart, and
            rescale the separation between axial slices to the standard 0.7mm spacing
        """
        self.abdo = BodyPartLoader(root_dir, "Abdo", rescale_on_load=rescale_on_load)
        # default to single body part for simplicity
        # self.thorax = BodyPartLoader(root_dir, 'Thorax', rescale_on_load=rescale_on_load)


class MultiPatientLoader:
    """A class providing intuitive loading of DICOM data patient-wise, scan-order-wise and body-region-wise

    Attributes:
        root_dir (pathlib Path): Path to the top level directory that stores multiple patient scans
        patient_paths (list of pathlib Paths): A natsorted list of paths to directories of patients, where each patient
                                               is expected to be stored in a purely numeric directory name
        patients (list of PatientLoaders): A list of PatientLoaders, one per patient in patient_paths, with the same
                                           ordering
    """

    def __init__(self, data_directory=None, rescale_on_load=True):
        """
        Build a data loader that seeks out patients, their body parts, and sequential scans within a directory with an
        expected structure

        Args:
            data_directory (pathlib Path): path to data directory, otherwise assume default extra data root directory
            rescale_on_load (bool): Whether to rescale axial planes such that each pixel is laterally 1mm apart, and
            rescale the separation between axial slices to the standard 0.7mm spacing
        """
        if data_directory is not None:
            self.root_dir = data_directory
        else:
            self.root_dir = data_root_directory()
        # load all folders and files not ending .txt
        patient_paths = list(self.root_dir.glob("[!.DS_Store]*"))
        disordered_stems = [int(path.stem) for path in patient_paths]
        sorted_path_order = np.argsort(disordered_stems)
        self.patient_paths = [patient_paths[i] for i in sorted_path_order]
        self.patients = [
            PatientLoader(path, rescale_on_load=rescale_on_load)
            for path in self.patient_paths
        ]


class MultiPatientAxialStreamer:
    """Class for streaming either random or sequential axial images from the CT dataset"""

    def __init__(self):
        self.multi_patient_loader = MultiPatientLoader()
        # default to single body part for simplicity
        # self.body_parts = ['abdo', 'thorax']
        self.body_parts = ["abdo"]
        self.scan_nums = ["scan_1", "scan_2"]
        self.curr_patient_index = 0
        self.curr_body_part_index = 0
        self.curr_scan_num_index = 0
        self.curr_axial_index = 0

    def reset_indices(self):
        """Set all the indices back to zero - useful for switching from random to sequential streaming"""
        self.curr_patient_index = 0
        self.curr_body_part_index = 0
        self.curr_scan_num_index = 0
        self.curr_axial_index = 0

    def stream_next(self, threshold=None, random=False):
        """Get the next image from the streamer

        Args:
            threshold (int or None): Minimum value within a streamed image to accept the streamed image. If this
            threshold isn't met, move on to the next image in the stream until one is found that has pixels above
            this thres. Useful for ignoring all-air images in the dataset with threshold=500
            random (bool): Whether to randomly select images from the dataset (True) or step through sequentially
            (False)

        Returns:
            (np.ndarray): 2D image

        """

        accept_scan = False

        while not accept_scan:
            if random is True:
                self.curr_patient_index = np.random.randint(
                    0, high=len(self.multi_patient_loader.patients)
                )
            curr_patient = self.multi_patient_loader.patients[self.curr_patient_index]

            if random is True:
                self.curr_body_part_index = np.random.randint(
                    0, high=len(self.body_parts)
                )
                self.curr_scan_num_index = np.random.randint(
                    0, high=len(self.scan_nums)
                )
            curr_body_part = curr_patient.__getattribute__(
                self.body_parts[self.curr_body_part_index]
            )
            curr_scan = curr_body_part.__getattribute__(
                self.scan_nums[self.curr_scan_num_index]
            )

            if curr_scan.rescaled_transverse_shape is None:
                (
                    curr_scan.rescaled_transverse_shape,
                    curr_scan.pixel_spacing,
                ) = curr_scan.raw_transverse_pixel_spacing_and_shape()

                if curr_scan.rescale_on_load:
                    curr_scan.rescaled_transverse_shape = tuple(
                        np.round(
                            np.array(curr_scan.rescaled_transverse_shape)
                            * curr_scan.pixel_spacing
                        ).astype(int)
                    )

            if random is True:
                self.curr_axial_index = np.random.randint(
                    0, high=len(curr_scan.dicom_paths)
                )
            out_im = curr_scan.load_2d_array(
                index=self.curr_axial_index, ignore_z_rescale=True
            )

            # iterate the slice. If reached the end of the axial length of the scan, iterate the scan number. If reached
            # the end of the scan number, itereate the body part, and so on. Allow returning to the beginning (ie first
            # patient, first scan, first body part, first axial slice
            if random is False:
                # only bother iterating if streaming is non-random
                self.curr_axial_index += 1
                if self.curr_axial_index == len(curr_scan.dicom_paths):
                    self.curr_axial_index = 0
                    self.curr_scan_num_index += 1
                    if self.curr_scan_num_index == len(self.scan_nums):
                        self.curr_scan_num_index = 0
                        self.curr_body_part_index += 1
                        if self.curr_body_part_index == len(self.body_parts):
                            self.curr_body_part_index = 0
                            self.curr_patient_index += 1
                            if self.curr_patient_index == len(
                                self.multi_patient_loader.patients
                            ):
                                self.curr_patient_index = 0

            if threshold is not None and out_im.max() >= threshold:
                accept_scan = True
            elif threshold is None:
                accept_scan = True
        if threshold is not None:
            out_im[out_im < threshold] = 0

        return out_im


def load_validation_set():
    path = data_root_directory().parent / "lesion_example" / "Lesion_example.csv"
    out_dict = {}
    try:
        df = read_csv(path)
        for index, row in df.iterrows():
            try:
                axial = int(row["Slice"])
            except ValueError:
                continue
            try:
                sagittal = int(row["X"])
            except ValueError:
                continue
            if sagittal == -1:
                continue
            try:
                coronal = int(row["Y"])
            except ValueError:
                continue
            if coronal == -1:
                continue
            patient = row["Patient"]
            if patient not in out_dict:
                out_dict[patient] = {}
            bodypart = row["Scan"][:-1].lower()
            if bodypart not in out_dict[patient]:
                out_dict[patient][bodypart] = {}
            scan_num = f'scan_{int(row["Scan"][-1])}'
            if scan_num not in out_dict[patient][bodypart]:
                out_dict[patient][bodypart][scan_num] = []
            out_dict[patient][bodypart][scan_num].append([axial, coronal, sagittal])
        return out_dict

    except FileNotFoundError:
        print(f"No validation data found, should be at {str(path)}")
