from enum import Enum

import cv2
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from PySide2 import QtCore
from PySide2.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QMessageBox,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
    QSlider,
    QSplitter,
    QLabel,
    QRadioButton,
    QSpinBox,
    QPushButton,
    QButtonGroup,
    QTableWidget,
    QTableWidgetItem,
    QAbstractItemView,
)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from ai_ct_scans import phase_correlation
from ai_ct_scans.image_processing_utils import normalise, overlay_warp_on_slice
from ai_ct_scans.keypoint_alignment import align_image
from ai_ct_scans.non_rigid_alignment import (
    transform_3d_volume_in_chunks,
    get_warp_overlay,
)
from ai_ct_scans.sectioning import CTEllipsoidFitter
from ai_ct_scans.phase_correlation_image_processing import generate_overlay_2d
from .image_viewer import ImageViewer
from .scan_viewer import ScanViewer


class SliceDirection(Enum):
    """Enum assigning identifier to slice orientation."""

    AXIAL = 0
    CORONAL = 1
    SAGITTAL = 2


class ViewerWidget(QWidget):
    """Widget to handle displaying scan data.

    Attributes:
        ui (ViewerWidget.Ui): UI class containing UI elements for this widget.
        data (dict): data for display.
        blank_frames (dict of np.ndarray): blank frame for each scan of correct size.
        phase_correlation_shift (tuple of int): Phase correlation shift values in each of 3 axes.
        orientation (SliceDirection): Orientation of scan view slicing.
        local_region (list of int): Local region for phase correlation alignment. 2D coordinates.
        cpd_aligned_scan (np.ndarray): Aligned 3D scan (using loaded CPD alignment)
    """

    class Ui:
        def __init__(self, widget, number_of_scans=0):
            """UI Class to contain UI elements for the ViewerWidget widget class.

            Args:
                widget (QWidget): parent widget to apply UI elements to.
            """
            widget.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )

            self.master_layout = QHBoxLayout()
            widget.setLayout(self.master_layout)

            self.main_widget = QWidget()

            self.layout = QVBoxLayout()
            self.main_widget.setLayout(self.layout)
            self.master_layout.addWidget(self.main_widget, stretch=8)

            # Make splitter
            self.splitter = QSplitter()
            self.splitter.setSizes([1, 1])
            self.layout.addWidget(self.splitter, stretch=8)

            # Create button widget for selecting orientation
            self.orientation_widget = QWidget()
            self.orientation_widget.setSizePolicy(
                QSizePolicy.Minimum, QSizePolicy.Minimum
            )
            self.orientation_layout = QHBoxLayout()
            self.orientation_widget.setLayout(self.orientation_layout)

            self.orientation_label = QLabel()
            self.orientation_label.setSizePolicy(
                QSizePolicy.Minimum, QSizePolicy.Minimum
            )
            self.orientation_label.setText("Slice Orientation")
            self.orientation_layout.addWidget(self.orientation_label, stretch=1)

            self.orientation_selection = QComboBox(widget)
            self.orientation_selection.addItems(
                [str(orientation).split(".")[1] for orientation in list(SliceDirection)]
            )
            self.orientation_layout.addWidget(self.orientation_selection, stretch=5)

            self.layout.addWidget(self.orientation_widget, stretch=1)

            # Create navigation slider
            self.slice_slider = QSlider(QtCore.Qt.Horizontal)
            self.slice_slider.setTickPosition(QSlider.TicksBothSides)
            self.slice_slider.setTickInterval(1)
            self.layout.addWidget(self.slice_slider, stretch=1)

            # Create controls for drawing alignment
            self.align_button_layout = QHBoxLayout()
            self.alignment_buttons = []
            self.alignment_button_group = QButtonGroup()
            self.unaligned_button = QRadioButton("Unaligned")
            self.unaligned_button.setChecked(True)
            self.alignment_buttons.append(self.unaligned_button)
            self.sift_button = QRadioButton("Keypoint (SIFT)")
            self.alignment_buttons.append(self.sift_button)
            self.orb_button = QRadioButton("Keypoint (ORB)")
            self.alignment_buttons.append(self.orb_button)
            self.phase_correlation_button = QRadioButton("Phase Correlation")
            self.alignment_buttons.append(self.phase_correlation_button)
            self.non_rigid_2d_button = QRadioButton("CPD")
            self.alignment_buttons.append(self.non_rigid_2d_button)
            for button in self.alignment_buttons:
                self.align_button_layout.addWidget(button)
                self.alignment_button_group.addButton(button)
            self.local_region_x_label = QLabel()
            self.local_region_x_label.setText("local region x")
            self.align_button_layout.addWidget(self.local_region_x_label)
            self.local_region_x = QSpinBox()
            self.local_region_x.setMaximum(1000)
            self.align_button_layout.addWidget(self.local_region_x)
            self.local_region_y_label = QLabel()
            self.local_region_y_label.setText("local region y")
            self.align_button_layout.addWidget(self.local_region_y_label)
            self.local_region_y = QSpinBox()
            self.local_region_y.setMaximum(1000)
            self.align_button_layout.addWidget(self.local_region_y)
            self.phase_correlation_push_button = QPushButton("Start")
            self.align_button_layout.addWidget(self.phase_correlation_push_button)

            self.layout.addLayout(self.align_button_layout)

            # For displaying sliced views
            self.slice_widget = QWidget()
            self.slice_widget.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )

            # Slice layout
            self.slice_layout = QVBoxLayout()

            # Create image views
            self.image_viewers = []
            for i in range(number_of_scans):
                self.image_viewers.append(ImageViewer())
            for viewer in self.image_viewers:
                self.slice_layout.addWidget(viewer)

            self.slice_widget.setLayout(self.slice_layout)

            # For displaying 3d or aligned results views
            self.view_widget = QWidget()
            self.view_widget.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )

            # Create results display
            self.results_display = ImageViewer()

            # Create non-rigid alignment warp display
            self.warp_widget = QWidget()
            self.warp_widget.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )
            self.warp_display = ImageViewer()
            self.warp_display.setVisible(False)
            self.warp_colour_bar_figure, self.warp_colour_bar_ax = plt.subplots(
                figsize=(0.35, 10), tight_layout=True
            )
            self.warp_colour_bar_figure.set_facecolor("None")
            self.warp_colour_bar = FigureCanvas(self.warp_colour_bar_figure)
            self.warp_colour_bar.setStyleSheet("background-color:transparent;")
            self.warp_colour_bar.setVisible(False)

            # Create non-rigid alignment warp display
            self.tissue_sectioning_widget = QWidget()
            self.tissue_sectioning_widget.setSizePolicy(
                QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
            )
            self.tissue_sectioning_display = ImageViewer()
            self.tissue_sectioning_display.setVisible(False)

            self.scan_display = ScanViewer()
            self.scan_display.setVisible(False)

            # Create 3d scan controls widget
            self.scan_control_layout = QHBoxLayout()

            self.scan_display_toggle = QCheckBox("Toggle 3D Scan View", widget)
            self.scan_control_layout.addWidget(self.scan_display_toggle)

            self.update_scan_button = QPushButton("Update 3D Scan View", widget)
            self.update_scan_button.setVisible(False)
            self.update_scan_button.setEnabled(False)
            self.scan_control_layout.addWidget(self.update_scan_button)

            self.layout.addLayout(self.scan_control_layout)
            self.scan_display_toggle.setChecked(False)

            # Tissue sectioning controls
            self.tissue_sectioning_control_layout = QHBoxLayout()
            self.tissue_sectioning_checkbox = QCheckBox("Enable tissue sectioning")
            self.tissue_sectioning_control_layout.addWidget(
                self.tissue_sectioning_checkbox
            )
            self.sectioning_scan_button_group = QButtonGroup()
            self.sectioning_scan_1_button = QRadioButton("Scan 1")
            self.tissue_sectioning_control_layout.addWidget(
                self.sectioning_scan_1_button
            )
            self.sectioning_scan_button_group.addButton(self.sectioning_scan_1_button)
            self.sectioning_scan_1_button.setChecked(True)
            self.sectioning_scan_2_button = QRadioButton("Scan 2")
            self.tissue_sectioning_control_layout.addWidget(
                self.sectioning_scan_2_button
            )
            self.sectioning_scan_button_group.addButton(self.sectioning_scan_2_button)
            self.layout.addLayout(self.tissue_sectioning_control_layout)

            # Ellipse Fitting Controls
            self.ellipse_control_layout = QHBoxLayout()
            self.ellipse_processing_button = QPushButton("Detect Ellipses")
            self.ellipse_control_layout.addWidget(self.ellipse_processing_button)
            self.layout.addLayout(self.ellipse_control_layout)
            self.ellipse_table_toggle = QPushButton("Show/Hide Ellipse Table")
            self.ellipse_control_layout.addWidget(self.ellipse_table_toggle)
            self.ellipse_table_toggle.hide()

            # Results view layout
            self.view_layout = QHBoxLayout()
            self.view_layout.addWidget(self.results_display)
            self.view_widget.setLayout(self.view_layout)

            # Warp view layout
            self.warp_layout = QHBoxLayout()
            self.warp_layout.addWidget(self.warp_display)
            self.warp_layout.addWidget(self.warp_colour_bar)
            self.warp_widget.setLayout(self.warp_layout)
            self.warp_widget.hide()

            # Tissue sectioning view layout
            self.tissue_sectioning_layout = QHBoxLayout()
            self.tissue_sectioning_layout.addWidget(self.tissue_sectioning_display)
            self.tissue_sectioning_widget.setLayout(self.tissue_sectioning_layout)

            # Add widgets to splitter view
            self.splitter.addWidget(self.slice_widget)
            self.splitter.addWidget(self.view_widget)
            self.splitter.addWidget(self.warp_widget)
            self.splitter.addWidget(self.scan_display)
            self.splitter.addWidget(self.tissue_sectioning_widget)

            # Add tables for detected items
            self.detection_widget = QWidget()

            self.detection_layout = QVBoxLayout()
            self.detection_widget.setLayout(self.detection_layout)
            self.master_layout.addWidget(self.detection_widget, stretch=3)

            self.ellipse_table_label = QLabel("Ellipse Detection Results")
            self.ellipse_table_label.setSizePolicy(
                QSizePolicy.Minimum, QSizePolicy.Minimum
            )
            self.detection_layout.addWidget(self.ellipse_table_label)
            self.ellipse_table_label.setStyleSheet("font-weight: bold")

            self.detection_table_1_label = QLabel("Detected Areas in Scan 1")
            self.detection_table_1_label.setSizePolicy(
                QSizePolicy.Minimum, QSizePolicy.Minimum
            )
            detection_table_1 = QTableWidget(0, 4)
            detection_table_1.setSelectionBehavior(QAbstractItemView.SelectRows)
            detection_table_1.setSelectionMode(QAbstractItemView.SingleSelection)
            detection_table_1.setEditTriggers(QAbstractItemView.NoEditTriggers)

            self.detection_table_2_label = QLabel("Detected Areas in Scan 2")
            self.detection_table_2_label.setSizePolicy(
                QSizePolicy.Minimum, QSizePolicy.Minimum
            )
            detection_table_2 = QTableWidget(0, 4)
            detection_table_2.setSelectionBehavior(QAbstractItemView.SelectRows)
            detection_table_2.setSelectionMode(QAbstractItemView.SingleSelection)
            detection_table_2.setEditTriggers(QAbstractItemView.NoEditTriggers)

            self.detection_tables = [detection_table_1, detection_table_2]
            self.detection_layout.addWidget(self.detection_table_1_label)
            self.detection_layout.addWidget(self.detection_tables[0])
            self.detection_layout.addWidget(self.detection_table_2_label)
            self.detection_layout.addWidget(self.detection_tables[1])

            self.detection_widget.hide()

    def __init__(self, number_of_scans=0):
        super().__init__()
        self.ui = self.Ui(self, number_of_scans=number_of_scans)

        self.data = {}
        self.maxes = {}
        self.blank_frames = {}
        self.phase_correlation_shift = None
        self.cpd_aligned_scan = None
        self.warp_overlay = None
        self.scan_display_enabled = False
        self.ellipses = None
        self.selected_ellipsoid = {}

        self.orientation = SliceDirection.AXIAL
        self.local_region = None

        self.ui.local_region_x.setValue(255)
        self.ui.local_region_y.setValue(255)
        self.toggle_phase_correlation_controls()
        self.toggle_non_rigid_2d_controls()

        self.ui.slice_slider.valueChanged.connect(lambda x: self.display_slice(x))
        self.ui.slice_slider.valueChanged.connect(lambda x: self.display_results(x))
        self.ui.slice_slider.valueChanged.connect(
            lambda x: self.display_sectioned_slice(x)
        )

        for button in self.ui.alignment_buttons:
            button.toggled.connect(
                lambda x: self.display_results(self.ui.slice_slider.value())
            )
        self.ui.phase_correlation_button.toggled.connect(
            self.toggle_phase_correlation_controls
        )
        self.ui.phase_correlation_push_button.clicked.connect(
            self.show_phase_correlation
        )
        self.ui.non_rigid_2d_button.toggled.connect(self.toggle_warp_display)

        # Toggle and start button only needed if the non-rigid alignment takes a long time.
        self.ui.non_rigid_2d_button.toggled.connect(self.toggle_non_rigid_2d_controls)

        self.ui.local_region_x.valueChanged.connect(self.update_local_region)
        self.ui.local_region_y.valueChanged.connect(self.update_local_region)

        self.ui.orientation_selection.currentIndexChanged.connect(
            self.change_orientation
        )
        self.ui.orientation_selection.currentIndexChanged.connect(
            lambda x: self.ui.update_scan_button.setEnabled(True)
        )

        self.ui.slice_slider.valueChanged.connect(
            lambda x: self.ui.update_scan_button.setEnabled(True)
        )
        self.ui.non_rigid_2d_button.toggled.connect(
            lambda x: self.ui.update_scan_button.setEnabled(True)
        )
        self.ui.update_scan_button.clicked.connect(
            lambda: self.update_scan(self.ui.slice_slider.value())
        )

        self.ui.scan_display_toggle.stateChanged.connect(self.update_scan_toggle)
        self.ui.tissue_sectioning_checkbox.stateChanged.connect(
            self.toggle_tissue_sectioning
        )
        for button in [
            self.ui.sectioning_scan_1_button,
            self.ui.sectioning_scan_2_button,
        ]:
            button.toggled.connect(
                lambda x: self.display_sectioned_slice(self.ui.slice_slider.value())
            )

        self.ui.ellipse_processing_button.clicked.connect(self.detect_ellipses)
        self.ui.detection_tables[0].itemSelectionChanged.connect(
            lambda: self.handle_item_selection(0)
        )
        self.ui.detection_tables[1].itemSelectionChanged.connect(
            lambda: self.handle_item_selection(1)
        )
        self.ui.ellipse_table_toggle.clicked.connect(self.toggle_ellipse_table)

    # TODO: Deal with situation of no/partial data.
    def set_data(self, data, cpd_alignment=None, sectioning_model=None):
        """Set data for displaying. If CPD alignment is present, calculate the 3D transform for second scan.

        Args:
            data (dictionary of np.ndarray): dictionary with two or more full scan arrays
            cpd_alignment (sklearn.pipeline.Pipeline): CPD alignment pipeline object
            sectioning_model (sectioning.TextonSectioner): A trained tissue sectioning model
        """
        self.phase_correlation_shift = None
        self.sectioner = sectioning_model
        self.ui.unaligned_button.setChecked(True)

        self.toggle_ellipse_table_button()

        for label, raw_scan in data.items():
            scan = raw_scan.copy()
            self.maxes[label] = scan.max()
            scan *= 1.0 / scan.max()
            self.data[label] = scan

        self.setup_orientation(self.orientation)

        # Set cpd alignment
        if cpd_alignment is not None:
            self.cpd_aligned_scan = transform_3d_volume_in_chunks(
                self.data["scan_2"].copy(), cpd_alignment.predict, chunk_thickness=100
            )
            self.warp_overlay = get_warp_overlay(
                self.cpd_aligned_scan.shape, cpd_alignment.predict, chunk_thickness=100
            )
        self.toggle_non_rigid_2d_controls()

        self.display_slice(0)
        self.display_results(0)

        self.ui.scan_display.set_data(self.data)

    def setup_orientation(self, orientation):
        """Configure class attributes specific to the currently selected orientation.

        Args:
            orientation (SliceDirection): Enum value for slice orientation
        """
        for label, scan in self.data.items():
            self.blank_frames[label] = np.zeros(
                (
                    scan.take(indices=0, axis=orientation.value).shape[0],
                    scan.take(indices=0, axis=orientation.value).shape[1],
                )
            )

        self.ui.slice_slider.setMinimum(0)
        max_depth = max(
            [scan.shape[orientation.value] - 1 for _, scan in self.data.items()]
        )
        self.ui.slice_slider.setMaximum(max_depth)

        self.orientation = orientation

    def extract_ellipse_overlay_info(self, scan_label, slice_index):
        """Extract closest match for ellipse in current axis from ellipsoid
            contributing ellipses.

        Args:
            scan_label (str): string label for which scan to refer to. e.g. scan_1 or scan_2
            slice_index (int): current slice index.

        Returns:
            (dict): dictionary describing ellipse to overlay, or None if no matches in current
                    orientation.
        """
        overlay_ellipse = None
        if (
            scan_label in self.selected_ellipsoid.keys()
            and self.selected_ellipsoid[scan_label] is not None
        ):
            ellipse_raw_info = self.selected_ellipsoid[scan_label]

            # Filter by current axis (orientation)
            remaining_ellipses = list(
                filter(
                    lambda x: x[3] == self.orientation.value,
                    ellipse_raw_info["contributing_ellipses"],
                )
            )

            if remaining_ellipses == []:
                return None

            # filter by nearest slice
            slices = [x[0][self.orientation.value] for x in remaining_ellipses]
            closest_value = min(slices, key=lambda val: abs(val - slice_index))

            if abs(closest_value - slice_index) > 2:
                return None

            remaining_ellipse = remaining_ellipses[slices.index(closest_value)]

            # Format centre coordinates
            center = list(remaining_ellipse[0])
            # eliminate current axis - then it is y, x
            center.pop(self.orientation.value)

            # Extract centre coordinates, sizes and angle - swapping y, x to x, y for opencv image coordinates.
            overlay_ellipse = {
                "center": center[::-1],
                "axis": remaining_ellipse[1],
                "angle": remaining_ellipse[2],
            }

        return overlay_ellipse

    def display_slice(self, slice_index):
        """Display axial slice of full scan.

        Args:
            slice_index (int): slice index
        """
        # TODO: Read and pass through lesion data from either first or second scan selection.
        for index, (label, scan) in enumerate(self.data.items()):
            if slice_index < scan.shape[self.orientation.value]:
                ellipse_to_overlay = self.extract_ellipse_overlay_info(
                    label, slice_index
                )
                self.ui.image_viewers[index].set_image(
                    scan.take(indices=slice_index, axis=self.orientation.value),
                    f"Scan {index + 1}: Slice {slice_index}",
                    self.local_region,
                    ellipse_to_overlay,
                )
            else:
                self.ui.image_viewers[index].set_image(
                    self.blank_frames[label], f"Scan {index + 1}: Slice {slice_index}"
                )

    def display_results(self, slice_index):
        """Method to display results using the results display.

        Args:
            slice_index (int): slice index
        """
        if not self.data:
            return

        result_image = list(self.blank_frames.values())[0]
        results_label = "PLACEHOLDER"

        if slice_index >= self.data["scan_1"].shape[self.orientation.value]:
            results_label = "Scan 1 index out of bounds"
        elif slice_index >= self.data["scan_2"].shape[self.orientation.value]:
            results_label = "Scan 2 index out of bounds"
        else:
            scan_1 = self.data["scan_1"].take(
                indices=slice_index, axis=self.orientation.value
            )
            scan_2 = self.data["scan_2"].take(
                indices=slice_index, axis=self.orientation.value
            )
            try:
                if self.ui.unaligned_button.isChecked():
                    scan_2_aligned = self.data["scan_2"].take(
                        indices=slice_index, axis=self.orientation.value
                    )
                    results_label = "Unaligned"
                elif self.ui.sift_button.isChecked():
                    scan_2_aligned = align_image(
                        normalise(scan_2), normalise(scan_1), detector="SIFT"
                    )
                    results_label = "SIFT keypoint detection aligned"
                elif self.ui.orb_button.isChecked():
                    scan_2_aligned = align_image(
                        normalise(scan_2), normalise(scan_1), detector="ORB"
                    )
                    results_label = "ORB keypoint detection aligned"
                elif self.ui.phase_correlation_button.isChecked():
                    if self.phase_correlation_shift is None:
                        scan_2_aligned = self.data["scan_2"].take(
                            indices=slice_index, axis=self.orientation.value
                        )
                        results_label = (
                            "Unaligned: Phase correlation shift not calculated"
                        )
                    else:
                        scan_2_aligned = phase_correlation.shift_nd(
                            self.data["scan_2"], -self.phase_correlation_shift
                        ).take(indices=slice_index, axis=self.orientation.value)
                        results_label = "Phase correlation aligned"
                elif self.ui.non_rigid_2d_button.isChecked():
                    if self.cpd_aligned_scan is not None:
                        scan_2_aligned = self.cpd_aligned_scan.take(
                            indices=slice_index, axis=self.orientation.value
                        )
                        results_label = "CPD aligned."
                        warp_overlay_slice = self.warp_overlay.take(
                            indices=slice_index, axis=self.orientation.value
                        )
                        warp_image = overlay_warp_on_slice(
                            scan_2_aligned, warp_overlay_slice
                        )
                        self.ui.warp_display.set_image(warp_image, "CPD warp.")

                        # Create colourbar
                        normalize = mcolors.Normalize(
                            vmin=np.min(warp_overlay_slice),
                            vmax=np.max(warp_overlay_slice),
                        )
                        self.ui.warp_colour_bar_figure.colorbar(
                            cm.ScalarMappable(norm=normalize, cmap=cm.jet),
                            cax=self.ui.warp_colour_bar_ax,
                        )
                        self.ui.warp_colour_bar_ax.tick_params(labelsize=4)
                        self.ui.warp_colour_bar.draw()
                    else:
                        scan_2_aligned = self.data["scan_2"].take(
                            indices=slice_index, axis=self.orientation.value
                        )
                        results_label = "Unaligned: Non-rigid alignment not calculated."
                result_image = generate_overlay_2d([scan_1, scan_2_aligned])
            except cv2.error as error:
                # Inspect openCV error strings
                if "arrays should have at least 4 corresponding point" in error.err:
                    results_label = "Failed to match enough keypoints"
                else:
                    results_label = error.err

        self.ui.results_display.set_image(result_image, results_label)

    def update_scan_toggle(self):
        """Update scan display visibility depending on toggle value."""
        self.scan_display_enabled = self.ui.scan_display_toggle.isChecked()
        self.ui.scan_display.setVisible(self.scan_display_enabled)
        self.update_scan(self.ui.slice_slider.value())
        self.ui.update_scan_button.setVisible(self.scan_display_enabled)

        sizes = []
        if self.scan_display_enabled:
            sizes = [1, 4, 4, 4, 4]
        else:
            sizes = [1, 4, 4, 4, 0]
        for i in range(self.ui.splitter.count()):
            self.ui.splitter.setStretchFactor(i, sizes[i])

    def update_scan(self, slice_index):
        """Update 3D scan view with updated slice index or orientation."""
        if self.scan_display_enabled:
            if self.ui.non_rigid_2d_button.isChecked():
                if self.cpd_aligned_scan is not None:
                    data = self.data
                    data["scan_2"] = self.cpd_aligned_scan
                    self.ui.scan_display.set_data(data)
            self.ui.scan_display.display_data(
                slice_index,
                self.orientation.value,
                slice_through_scan=False,
                show_navigation_slice=True,
            )
        self.ui.update_scan_button.setEnabled(False)

    def change_orientation(self, index):
        """Update orientation setting and displays.

        Args:
            index (int): index of orientation to change to
        """
        self.setup_orientation(SliceDirection(index))

        self.display_slice(self.ui.slice_slider.value())
        self.display_results(self.ui.slice_slider.value())
        self.display_sectioned_slice(self.ui.slice_slider.value())
        if self.scan_display_enabled:
            self.update_scan(self.ui.slice_slider.value())

    def toggle_phase_correlation_controls(self):
        """Hide or show the controls used for phase correlation alignment."""
        if self.ui.phase_correlation_button.isChecked():
            self.ui.local_region_x.show()
            self.ui.local_region_y.show()
            self.ui.local_region_x_label.show()
            self.ui.local_region_y_label.show()
            self.ui.phase_correlation_push_button.show()
            self.local_region = (
                int(self.ui.local_region_x.value()),
                int(self.ui.local_region_y.value()),
            )
        else:
            self.ui.local_region_x.hide()
            self.ui.local_region_y.hide()
            self.ui.local_region_x_label.hide()
            self.ui.local_region_y_label.hide()
            self.ui.phase_correlation_push_button.hide()
            self.local_region = None

        if self.data:
            self.display_slice(self.ui.slice_slider.value())

    def toggle_non_rigid_2d_controls(self):
        """Hide or show the controls used for non-rigid alignment."""
        # Only required if the non-rigid alignment takes a long time and needs to be triggered by the user.
        if self.cpd_aligned_scan is None:
            self.ui.non_rigid_2d_button.hide()
        else:
            self.ui.non_rigid_2d_button.show()

    def update_local_region(self):
        """Update the local region that has been defined for phase correlation alignment, as (x, y) coordinates."""
        self.local_region = [
            int(self.ui.local_region_x.value()),
            int(self.ui.local_region_y.value()),
        ]
        self.display_slice(self.ui.slice_slider.value())

    # TODO: Test that scan display  and results display is updated with phase shift
    def show_phase_correlation(self):
        """Perform phase correlation based around the local region defined."""
        slice_index = self.ui.slice_slider.value()
        scan_1 = self.data["scan_1"].take(
            indices=slice_index, axis=self.orientation.value
        )

        if slice_index < self.data["scan_1"].shape[self.orientation.value]:
            full_views = [self.data["scan_1"], self.data["scan_2"]]
            local_coords = [self.local_region[1], self.local_region[0]]
            local_coords.insert(self.orientation.value, slice_index)
            shifts = phase_correlation.shifts_via_local_region(
                full_views,
                local_coords=local_coords,
                region_widths=(100, 100, 100),
                apply_lmr=True,
                apply_zero_crossings=True,
                lmr_radius=3,
            )

            aligned_scan = phase_correlation.shift_nd(full_views[1], -shifts[1])
            scan_2_aligned = aligned_scan.take(
                indices=slice_index, axis=self.orientation.value
            )
            self.phase_correlation_shift = shifts[1]
            self.ui.results_display.set_image(
                generate_overlay_2d([scan_1, scan_2_aligned])
            )

            if self.scan_display_enabled:
                if self.phase_correlation_shift is not None:
                    data = self.data
                    data["scan_2"] = aligned_scan
                    self.ui.scan_display.set_data(data)

                self.ui.scan_display.display_data(
                    slice_index,
                    self.orientation.value,
                    slice_through_scan=False,
                    show_navigation_slice=True,
                )

    def calculate_non_rigid_alignment(self):
        """Calculate the non-rigid alignment, store the offset and trigger an update to the display."""
        # TODO: Use this method to calculate non-rigid alignment if unable to do so live.
        pass

    def toggle_warp_display(self):
        """Display or hide the warp display."""
        self.ui.warp_widget.setVisible(self.ui.non_rigid_2d_button.isChecked())
        self.ui.warp_display.setVisible(self.ui.non_rigid_2d_button.isChecked())
        self.ui.warp_colour_bar.setVisible(self.ui.non_rigid_2d_button.isChecked())

    def toggle_tissue_sectioning(self):
        """Display or hide the tissue sectioning controls and display."""
        self.ui.tissue_sectioning_display.setVisible(
            self.ui.tissue_sectioning_checkbox.isChecked()
        )
        self.display_sectioned_slice(self.ui.slice_slider.value())

    def display_sectioned_slice(self, slice_index):
        """Method to display results of tissue sectioning.

        Args:
            slice_index (int): slice index
        """
        scan_label = (
            "scan_1" if self.ui.sectioning_scan_1_button.isChecked() else "scan_2"
        )
        if self.ui.tissue_sectioning_checkbox.isChecked():
            if self.sectioner is None:
                self.ui.tissue_sectioning_display.set_image(
                    list(self.blank_frames.values())[0], "No sectioning model loaded"
                )
                return
            scan = (
                self.data[scan_label].take(
                    indices=slice_index, axis=self.orientation.value
                )
                * self.maxes[scan_label]
            )
            sectioned_image = (
                self.sectioner.label_im(
                    scan, threshold=500, clusterer_ind=0, full_sub_structure=True
                )
                + 1
            ).astype("uint8")
            sectioned_image = cv2.applyColorMap(
                normalise(cv2.equalizeHist(sectioned_image).astype("uint8")),
                cv2.COLORMAP_JET,
            )
            self.ui.tissue_sectioning_display.set_image(
                normalise(sectioned_image), scan_label
            )

    def detect_ellipses(self):
        """Method for triggering ellipse detection on each scan."""
        ellipsoid_fitter = CTEllipsoidFitter(
            min_ellipse_long_axis=15, max_ellipse_long_axis=200, max_area=25000
        )

        if self.sectioner is None or self.sectioner.clusterers is None:
            _, scan_1_ellipses = ellipsoid_fitter.draw_ellipsoid_walls(
                self.data["scan_1"] * self.maxes["scan_1"]
            )
            _, scan_2_ellipses = ellipsoid_fitter.draw_ellipsoid_walls(
                self.data["scan_2"] * self.maxes["scan_2"]
            )
        else:
            _, scan_1_ellipses = ellipsoid_fitter.draw_ellipsoid_walls(
                self.data["scan_1"] * self.maxes["scan_1"],
                sectioner=self.sectioner,
                sectioner_kwargs={
                    "threshold": 500,
                    "clusterer_ind": 0,
                    "full_sub_structure": True,
                },
            )
            _, scan_2_ellipses = ellipsoid_fitter.draw_ellipsoid_walls(
                self.data["scan_2"] * self.maxes["scan_2"],
                sectioner=self.sectioner,
                sectioner_kwargs={
                    "threshold": 500,
                    "clusterer_ind": 0,
                    "full_sub_structure": True,
                },
            )

        self._display_info(
            "Finished Ellipse Detection",
            f"Scan 1 detected: {len(scan_1_ellipses)} ellipses\n"
            + f"Scan 2 detected: {len(scan_2_ellipses)} ellipses",
        )

        sorted_scan_1_ellipses = sorted(
            scan_1_ellipses, key=lambda k: k["volume"], reverse=True
        )
        sorted_scan_2_ellipses = sorted(
            scan_2_ellipses, key=lambda k: k["volume"], reverse=True
        )

        self.update_ellipse_table(sorted_scan_1_ellipses, sorted_scan_2_ellipses)

        self.ellipses = {
            "scan_1": sorted_scan_1_ellipses,
            "scan_2": sorted_scan_2_ellipses,
        }
        self.toggle_ellipse_table_button()

        self.display_slice(self.ui.slice_slider.value())

    def update_ellipse_table(self, scan_1_ellipses=None, scan_2_ellipses=None):
        """Method for updating the detection table with ellipse data.

        Args:
            scan_1_ellipses (list): List of ellipse dictionaries for scan 1
            scan_2_ellipses (list): List of ellipse dictionaries for scan 2
        """
        if scan_1_ellipses is not None and scan_2_ellipses is not None:
            id = 0
            self.ui.detection_tables[0].setRowCount(len(scan_1_ellipses))
            self.ui.detection_tables[1].setRowCount(len(scan_2_ellipses))
            self.ui.detection_tables[0].setHorizontalHeaderLabels(
                ["Lesion ID", "Centre", "Volume", "Class"]
            )
            self.ui.detection_tables[1].setHorizontalHeaderLabels(
                ["Lesion ID", "Centre", "Volume", "Class"]
            )
            for i, item in enumerate(scan_1_ellipses):
                if item["centre"] is not None:
                    centre_string = f"[{item['centre'][0]:.2f} {item['centre'][1]:.2f} {item['centre'][2]:.2f}]"
                else:
                    centre_string = ""
                self.ui.detection_tables[0].setItem(i, 0, QTableWidgetItem(str(id)))
                self.ui.detection_tables[0].setItem(
                    i, 1, QTableWidgetItem(centre_string)
                )
                self.ui.detection_tables[0].setItem(
                    i, 2, QTableWidgetItem(f"{item['volume']:.2f}")
                )
                self.ui.detection_tables[0].setItem(
                    i, 3, QTableWidgetItem(str(item["class"]))
                )
                id += 1
            for i, item in enumerate(scan_2_ellipses):
                if item["centre"] is not None:
                    centre_string = f"[{item['centre'][0]:.2f} {item['centre'][1]:.2f} {item['centre'][2]:.2f}]"
                else:
                    centre_string = ""
                self.ui.detection_tables[1].setItem(i, 0, QTableWidgetItem(str(id)))
                self.ui.detection_tables[1].setItem(
                    i, 1, QTableWidgetItem(centre_string)
                )
                self.ui.detection_tables[1].setItem(
                    i, 2, QTableWidgetItem(f"{item['volume']:.2f}")
                )
                self.ui.detection_tables[1].setItem(
                    i, 3, QTableWidgetItem(str(item["class"]))
                )
                id += 1

            self.ui.detection_widget.show()

    def handle_item_selection(self, table_index):
        """Detect the selected table item in the scan ellipse table
        and move the slice view to the relevant slice.
        """
        indices = self.ui.detection_tables[table_index].selectedItems()

        if indices:
            if table_index == 0:
                self.ui.detection_tables[1].clearSelection()
            else:
                self.ui.detection_tables[0].clearSelection()

            split_string = self.extract_centre_coordinates_from_string(
                indices[1].text()
            )
            int_values = [round(x) for x in split_string]

            # Get index of row
            rows = [
                idx.row()
                for idx in self.ui.detection_tables[table_index].selectedIndexes()
            ]
            row = list(set(rows))[0]

            # Get data from ellipse dictionary
            selected_ellipsoid = self.ellipses[f"scan_{table_index + 1}"][row]

            # Store selected ellipsoid data for display
            self.selected_ellipsoid = {"scan_1": None, "scan_2": None}
            self.selected_ellipsoid[f"scan_{table_index + 1}"] = selected_ellipsoid

            self.ui.slice_slider.setValue(int_values[self.orientation.value])

    def extract_centre_coordinates_from_string(self, info_string):
        """Extract slice coordinates as floats from string data in form "[x , y, z]".
            Removes any leading or trailing spaces inserted by PyQtTableWidgets.

        Args:
            info_string (string): list values formatted as single string.

        Returns:
            (list): list of centre coordinates from string.
        """
        trimmed_string = info_string[1:-1].strip()
        split_string = trimmed_string.split(" ")

        return [float(x.strip()) for x in split_string if x.strip()]

    def toggle_ellipse_table_button(self):
        """Programatically hide or show detection table toggle button."""
        if self.ellipses is None:
            self.ui.ellipse_table_toggle.hide()
        else:
            self.ui.ellipse_table_toggle.show()

    def toggle_ellipse_table(self):
        """If toggle button pressed, either hide or show detection table."""
        if self.ellipses is None:
            self.ui.detection_widget.hide()
        else:
            if self.ui.detection_widget.isHidden() is True:
                self.ui.detection_widget.show()
            else:
                self.ui.detection_widget.hide()

    def _display_info(self, title, info):
        """Method to generate a QMessageBox with provided string information.

        Args:
            title (str): title of message box.
            info (str): contents of message box.
        """
        QMessageBox.information(self, title, info)
