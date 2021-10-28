import numpy as np
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide2.QtWidgets import QWidget, QVBoxLayout, QSizePolicy


# TODO: Generate suitable scan colours for more than 2 scans.
SCAN_COLOURS = [
    (1, 0, 0, 0.4),
    (0, 0, 1, 0.4),
]


class ScanViewer(QWidget):
    """Widget for displaying 3D scan data.

    Attributes:
        volume_plot (pyqtgraph.opengl.GLVolumeItem): OpenGL 3D volume plot item
        viewer_data (np.ndarray): preprocessed array of scan data in format (x, y, z, RGBA)
    """

    class Ui:
        def __init__(self, widget):
            """UI Class to contain UI elements for the ViewerWidget widget class.

            Args:
                widget (QWidget): parent widget to apply UI elements to.
            """
            widget.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )

            self.layout = QVBoxLayout()
            widget.setLayout(self.layout)

            pg.setConfigOptions(antialias=True)

            self.viewer = gl.GLViewWidget()
            self.layout.addWidget(self.viewer)

    def __init__(self):
        super().__init__()
        self.ui = self.Ui(self)

        self.volume_plot = None
        self.viewer_data = None

    def set_data(self, data):
        """Preprocess scan data and add to OpenGL plot item.

        Args:
            data (dict): dictionary of np.ndarray, labelled 'scan_1', 'scan_2 for each full scan item.
        """
        self.viewer_data = self.pre_process_data(data)

        if self.viewer_data is not None:
            if self.volume_plot is not None:
                self.ui.viewer.removeItem(self.volume_plot)

            self.volume_plot = pg.opengl.GLVolumeItem(
                self.viewer_data, smooth=False, glOptions="translucent"
            )

            # Translate view to allow rotation around centre of plot
            self.volume_plot.translate(
                -self.viewer_data.shape[0] / 2,
                -self.viewer_data.shape[1] / 2,
                -self.viewer_data.shape[2] / 2,
            )

            # TODO: Initialise the camera viewpoint sensibly to show whole scan.
            # This initialises the zoom level but not the viewpoint.
            self.ui.viewer.setCameraPosition(distance=900)

            self.ui.viewer.addItem(self.volume_plot)

    def display_data(
        self,
        slice_index=0,
        orientation=0,
        slice_through_scan=False,
        show_navigation_slice=False,
    ):
        """Update display to show scan with specific slicing or overlay.

        Args:
            slice_index (int): Index of slice used to display or augment the displayed data.
            orientation (int): Slice orientation value.
            slice_through_scan (bool): Flag for specifying showing the cross section of the scan.
            show_navigation_slice (bool): Flag for adding the current slice location as plane to scan.
        """
        if self.viewer_data is not None:
            # Slice through scan, or add navigation plane to scan.
            if slice_through_scan:
                if orientation == 0:
                    self.volume_plot.setData(self.viewer_data[slice_index:, :, :, :])
                elif orientation == 1:
                    self.volume_plot.setData(self.viewer_data[:, slice_index:, :, :])
                elif orientation == 2:
                    self.volume_plot.setData(self.viewer_data[:, :, slice_index:, :])
            elif show_navigation_slice:
                slice_viewer_data = self.viewer_data.copy()
                if orientation == 0:
                    slice_viewer_data[slice_index, :, :, :] = [255, 255, 255, 64]
                elif orientation == 1:
                    slice_viewer_data[:, slice_index, :, :] = [255, 255, 255, 64]
                elif orientation == 2:
                    slice_viewer_data[:, :, slice_index, :] = [255, 255, 255, 64]
                self.volume_plot.setData(slice_viewer_data)
            else:
                self.volume_plot.setData(self.viewer_data)

    @staticmethod
    def pre_process_data(data):
        """Covert scan data into (x, y, z, RGBA) and combine into single structure for display.

        Args:
            data (dict): dictionary of np.ndarray, labelled 'scan_1', 'scan_2' for each full scan item.

        Returns:
            (np.ndarray): array of format (x, y, z, RGBA)
        """
        # Find max dimensions
        max_shape = [0, 0, 0]
        for scan in data.values():
            shape = scan.shape
            max_shape = np.where(list(shape) > max_shape, shape, max_shape)

        result = np.empty(list(max_shape).append(4), dtype=np.ubyte)
        for i, scan in enumerate(data.values()):
            # Pad scan to max_shape on ends of axis
            # For debugging: shape_difference = max_shape - list(scan.shape)
            padded_scan = np.zeros(max_shape)
            padded_scan[: scan.shape[0], : scan.shape[1], : scan.shape[2]] = scan

            # Loops backwards through the colours to ensure there is always a colour.
            result = np.add(
                result, ScanViewer.convert_for_viewer(padded_scan, SCAN_COLOURS[-i])
            )

        return result

    @staticmethod
    def convert_for_viewer(pointcloud, colour=(0, 0, 0, 0)):
        """Convert scan data into 3D pointcloud matrix of RGBA values for displaying.

        Args:
            pointcloud (np.ndarray): np.ndarray representing 3D scan data.
            colour (tuple): RGBA value to apply to pointcloud.

        Returns:
            (np.ndarray): array of format (x, y, z, RGBA) for display.
        """
        formatted_pointcloud = np.empty(pointcloud.shape + (4,), dtype=np.ubyte)

        intensity_vals = pointcloud * (255.0 / (pointcloud.max() / 1))

        # Set colour values
        formatted_pointcloud[..., 0] = intensity_vals * colour[0]
        formatted_pointcloud[..., 1] = intensity_vals * colour[1]
        formatted_pointcloud[..., 2] = intensity_vals * colour[2]

        # Initialise alpha intensity
        formatted_pointcloud[..., 3] = intensity_vals

        # Square raw intensity value to increase contrast and apply alpha transparency
        formatted_pointcloud[..., 3] = (
            (formatted_pointcloud[..., 3].astype(float) / 255.0) ** 2 * 255 * colour[3]
        )

        return formatted_pointcloud
