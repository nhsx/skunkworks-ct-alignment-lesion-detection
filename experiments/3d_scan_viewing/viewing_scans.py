import sys

import numpy as np
from PySide2.QtWidgets import QApplication
from PySide2 import QtCore
import pyqtgraph as pg
import pyqtgraph.opengl as gl

from ai_ct_scans.data_loading import MultiPatientLoader


def pre_process(scan):
    """Normalise and slice data."""

    scan *= 1.0 / scan.max()

    pointcloud = np.where(scan[:, :, :] > 0, scan[:, :, :], 0)
    print("Pointcloud shape: {}".format(pointcloud.shape))

    return pointcloud


def convert_for_viewer(pointcloud, colour):
    """Convert scan data into 3D pointcloud matrix of RGBA values for displaying."""
    points = pointcloud

    d2 = np.empty(points.shape + (4,), dtype=np.ubyte)

    intensity_vals = points * (255.0 / (points.max() / 1))

    d2[..., 0] = intensity_vals * colour[0]
    d2[..., 1] = intensity_vals * colour[1]
    d2[..., 2] = intensity_vals * colour[2]

    d2[..., 3] = intensity_vals

    # Square intensity value alpha to increase contrast.
    d2[..., 3] = (d2[..., 3].astype(float) / 255.0) ** 2 * 255 * colour[3]

    return d2


if __name__ == "__main__":
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling)
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps)

    # Initialise application
    app = QApplication([])
    pg.setConfigOptions(antialias=True)

    # Create OpenGL view widget
    view = gl.GLViewWidget()
    view.setWindowTitle("pyqtgraph: GLVolumeItem CT Scans")

    # Load scan data
    data = MultiPatientLoader()
    data.patients[0].abdo.scan_1.load_scan()
    print("Full scan 1 shape:{}".format(data.patients[0].abdo.scan_1.full_scan.shape))
    data.patients[0].abdo.scan_2.load_scan()
    print("Full scan 2 shape:{}".format(data.patients[0].abdo.scan_2.full_scan.shape))

    scan_1 = data.patients[0].abdo.scan_1.full_scan
    scan_2 = data.patients[0].abdo.scan_2.full_scan

    # Normalise scan data
    pointcloud_1 = pre_process(scan_1)
    pointcloud_2 = pre_process(scan_2)

    # Reshape data and assign RGBA values from intensity values for plotting
    viewer_data_1 = convert_for_viewer(pointcloud_1, (1, 0, 0, 0.1))
    viewer_data_2 = convert_for_viewer(
        pointcloud_2[: pointcloud_1.shape[0], :, :], (0, 0, 1, 0.1)
    )

    # Combine scan one and scan two
    viewer_data = np.add(viewer_data_1, viewer_data_2)

    # Add axis lines
    viewer_data[:, 0, 0] = [255, 0, 0, 255]
    viewer_data[0, :, 0] = [0, 255, 0, 255]
    viewer_data[0, 0, :] = [0, 0, 255, 255]

    # Create voxel volume plot item
    volume_plot = pg.opengl.GLVolumeItem(
        viewer_data, smooth=False, glOptions="translucent"
    )

    # Translate view to allow rotation around centre of mass of plot
    volume_plot.translate(
        -viewer_data_1.shape[0] / 2, -viewer_data_1.shape[1] / 2, -150
    )
    view.addItem(volume_plot)

    # Run application
    view.show()
    sys.exit(app.exec_())
