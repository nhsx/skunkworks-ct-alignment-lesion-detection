import datetime

import numpy as np
import pydicom
from pydicom.dataset import FileDataset, FileMetaDataset


def create_dicom_file(file_path, pixel_array, slice_location=1000.0):
    """Save a pixel array to a dicom file.

    Args:
        file_path (str): path to where you want to save the dicom file
        pixel_array (np.ndarray, uint16): The 2D data you wish to save in a dicom file
        slice_location (float): An absolute axial position for the slice

    Returns:
        (pydicom.dataset): pydicom.dataset with pixel_array accessible

    """
    # Only allowing uint16 writing
    pixel_array = pixel_array.astype("uint16")

    # Populate required values for file meta information
    file_meta = FileMetaDataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"
    file_meta.MediaStorageSOPInstanceUID = "1.2.3"
    file_meta.ImplementationClassUID = "1.2.3.4"
    file_meta.TransferSyntaxUID = pydicom.uid.ImplicitVRLittleEndian

    ds = FileDataset(str(file_path), {}, file_meta=file_meta, preamble=b"\0" * 128)

    # Add the data elements -- not trying to set all required here. Check DICOM
    # standard
    ds.PatientName = "NAME^NONE"
    ds.PatientID = "123"
    ds.PixelSpacing = [0.5, 0.75]

    # Set the transfer syntax
    ds.is_little_endian = True
    ds.is_implicit_VR = True

    # Set timestamp
    dt = datetime.datetime.now()
    ds.ContentDate = dt.strftime("%Y%m%d")
    ds.ContentTime = dt.strftime("%H%M%S")

    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.PixelRepresentation = 0
    ds.HighBit = 15
    ds.BitsAllocated = 16
    ds.BitsStored = 16
    ds.LargestImagePixelValue = 255
    ds.SmallestImagePixelValue = 0
    ds.SliceLocation = slice_location
    ds.Columns = pixel_array.shape[1]
    ds.Rows = pixel_array.shape[0]
    ds.PixelData = pixel_array.tobytes()

    ds.save_as(str(file_path))
    return ds


def ndarray_to_memmap(arr, directory, prefix):
    """Writes an ndarray to hard drive with a generated name that stores shape information to allow reloading
    Args:
        arr (ndarray): ndarray to be saved
        directory (pathlib Path): pathlib Path to a directory
        prefix (str): a filename, without file extensions

    Returns:
        (pathlib Path): the path to the saved memmap

    """
    # remove file extension if supplied
    prefix = prefix.split(".")[0]
    # generate the shape string
    shape_str = "_".join([str(element) for element in list(arr.shape)])
    # generate the full save path
    save_path = directory / f"{prefix}__{shape_str}.npy"
    memmap_arr = np.memmap(save_path, dtype=arr.dtype, mode="w+", shape=arr.shape)
    memmap_arr[:] = arr[:]
    del memmap_arr
    return save_path
