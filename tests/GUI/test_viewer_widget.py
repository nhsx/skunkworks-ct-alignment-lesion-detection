from enum import Enum

import pytest
import mock
import numpy as np
from PySide2.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QSlider,
    QSplitter,
    QLabel,
)
from PySide2 import QtCore

from ai_ct_scans.GUI import ViewerWidget, ImageViewer, ScanViewer
from ai_ct_scans.GUI.viewer_widget import SliceDirection
from ai_ct_scans.phase_correlation_image_processing import generate_overlay_2d, sphere
from ai_ct_scans.phase_correlation import shift_nd
from ai_ct_scans.keypoint_alignment import align_image


class TestSliceDirection:
    def test_slice_direction(self):
        assert issubclass(SliceDirection, Enum)

    def test_slice_direction_values(self):
        assert SliceDirection.AXIAL.value == 0
        assert SliceDirection.CORONAL.value == 1
        assert SliceDirection.SAGITTAL.value == 2

    def test_slice_direction_length(self):
        assert len(list(SliceDirection)) == 3


@pytest.fixture
def window_widget(qtbot):
    window = ViewerWidget(number_of_scans=2)
    qtbot.addWidget(window)
    return window


@pytest.fixture()
def sectioner_patch():
    patch = mock.MagicMock()
    patch.label_im.return_value = np.zeros([50, 50])
    return patch


class TestViewerWidgetUI:
    def test_ui_viewer_widget_is_qwidget(self):
        assert issubclass(ViewerWidget, QWidget)

    def test_ui_size_policy(self, window_widget):
        assert window_widget.sizePolicy() == QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

    def test_main_layout(self, window_widget):
        assert isinstance(window_widget.ui.layout, QVBoxLayout)

    def test_sets_layout(self, window_widget):
        assert window_widget.layout() == window_widget.ui.master_layout

    # Test splitter
    def test_splitter_widget(self, window_widget):
        assert isinstance(window_widget.ui.splitter, QSplitter)

    def test_splitter_added_to_layout(self, window_widget):
        assert window_widget.ui.layout.itemAt(0).widget() == window_widget.ui.splitter

    # Test orientation widget
    def test_orientation_widget(self, window_widget):
        assert isinstance(window_widget.ui.orientation_widget, QWidget)

    def test_orientation_widget_size(self, window_widget):
        assert window_widget.ui.orientation_widget.sizePolicy() == QSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Minimum
        )

    def test_add_orientation_widget_to_layout(self, window_widget):
        assert (
            window_widget.ui.layout.itemAt(1).widget()
            == window_widget.ui.orientation_widget
        )

    def test_orientation_layout(self, window_widget):
        assert isinstance(window_widget.ui.orientation_layout, QHBoxLayout)

    def test_sets_orientation_layout(self, window_widget):
        assert (
            window_widget.ui.orientation_widget.layout()
            == window_widget.ui.orientation_layout
        )

    # Test orientation label
    def test_orientation_label(self, window_widget):
        assert isinstance(window_widget.ui.orientation_label, QLabel)

    def test_orientation_label_size(self, window_widget):
        assert window_widget.ui.orientation_label.sizePolicy() == QSizePolicy(
            QSizePolicy.Minimum, QSizePolicy.Minimum
        )

    def test_orienation_label_text(self, window_widget):
        assert window_widget.ui.orientation_label.text() == "Slice Orientation"

    def test_orientation_label_added_to_layout(self, window_widget):
        assert (
            window_widget.ui.orientation_layout.itemAt(0).widget()
            == window_widget.ui.orientation_label
        )

    # Test orientation selection box
    def test_orientation_selection(self, window_widget):
        assert isinstance(window_widget.ui.orientation_selection, QComboBox)

    def test_orientation_selection_items(self, window_widget):
        expected = ["AXIAL", "CORONAL", "SAGITTAL"]
        assert window_widget.ui.orientation_selection.findText(expected[0]) == 0
        assert window_widget.ui.orientation_selection.findText(expected[1]) == 1
        assert window_widget.ui.orientation_selection.findText(expected[2]) == 2

    def test_add_orientation_selection_to_layout(self, window_widget):
        assert (
            window_widget.ui.orientation_layout.itemAt(1).widget()
            == window_widget.ui.orientation_selection
        )

    # Test navigation slider
    def test_initialise_slider(self, window_widget):
        assert isinstance(window_widget.ui.slice_slider, QSlider)

    def test_set_slider_tick_interval(self, window_widget):
        assert window_widget.ui.slice_slider.tickInterval() == 1

    def test_set_slider_tick_position(self, window_widget):
        assert window_widget.ui.slice_slider.tickPosition() == QSlider.TicksBothSides

    def test_add_slider_to_layout(self, window_widget):
        assert (
            window_widget.ui.layout.itemAt(2).widget() == window_widget.ui.slice_slider
        )

    def test_slider_orientation(self, window_widget):
        assert window_widget.ui.slice_slider.orientation() == QtCore.Qt.Horizontal

    def test_slider_minimum(self, window_widget):
        assert window_widget.ui.slice_slider.minimum() == 0

    def test_slider_maximum_default(self, window_widget):
        assert window_widget.ui.slice_slider.maximum() == 99

    # Test alignment radio buttons
    def test_unaligned_button_initialised_to_checked(self, window_widget):
        assert window_widget.ui.unaligned_button.isChecked() is True

    def test_sift_button_initialised_to_unchecked(self, window_widget):
        assert window_widget.ui.sift_button.isChecked() is False

    def test_orb_button_initialised_to_unchecked(self, window_widget):
        assert window_widget.ui.orb_button.isChecked() is False

    def test_phase_correlation_button_initialised_to_unchecked(self, window_widget):
        assert window_widget.ui.phase_correlation_button.isChecked() is False

    def test_radio_buttons_appear_in_layout(self, window_widget):
        assert (
            window_widget.ui.layout.itemAt(3).layout()
            == window_widget.ui.align_button_layout
        )
        assert (
            window_widget.ui.align_button_layout.itemAt(0).widget()
            == window_widget.ui.unaligned_button
        )
        assert (
            window_widget.ui.align_button_layout.itemAt(1).widget()
            == window_widget.ui.sift_button
        )
        assert (
            window_widget.ui.align_button_layout.itemAt(2).widget()
            == window_widget.ui.orb_button
        )
        assert (
            window_widget.ui.align_button_layout.itemAt(3).widget()
            == window_widget.ui.phase_correlation_button
        )

    def test_phase_correlation_buttons_initialised_to_hidden(self, window_widget):
        for widget in [
            window_widget.ui.local_region_x_label,
            window_widget.ui.local_region_x,
            window_widget.ui.local_region_y_label,
            window_widget.ui.local_region_y,
            window_widget.ui.phase_correlation_push_button,
        ]:
            assert widget.isHidden()

    def test_local_region_coordinates_default(self, window_widget):
        assert window_widget.ui.local_region_x.value() == 255
        assert window_widget.ui.local_region_y.value() == 255

    def test_local_region_coordinates_maximums_set_to_1000(self, window_widget):
        assert window_widget.ui.local_region_x.maximum() == 1000
        assert window_widget.ui.local_region_y.maximum() == 1000

    # Test slice widget
    def test_slice_widget(self, window_widget):
        assert isinstance(window_widget.ui.slice_widget, QWidget)

    def test_slice_widget_size(self, window_widget):
        assert window_widget.ui.slice_widget.sizePolicy() == QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

    # Test Image Slice Displays
    def test_ui_initialises_image_viewers(self, window_widget):
        for viewer in window_widget.ui.image_viewers:
            assert isinstance(viewer, ImageViewer)

    def test_sets_slice_layout(self, window_widget):
        assert isinstance(window_widget.ui.slice_layout, QVBoxLayout)

    def test_add_slice_layout_to_slice_widget(self, window_widget):
        assert window_widget.ui.slice_widget.layout() == window_widget.ui.slice_layout

    def test_adds_display_1_to_layout(self, window_widget):
        assert (
            window_widget.ui.slice_layout.itemAt(0).widget()
            == window_widget.ui.image_viewers[0]
        )

    def test_adds_display_2_to_layout(self, window_widget):
        assert (
            window_widget.ui.slice_layout.itemAt(1).widget()
            == window_widget.ui.image_viewers[1]
        )

    # Test view widget
    def test_view_widget(self, window_widget):
        assert isinstance(window_widget.ui.view_widget, QWidget)

    def test_view_widget_size(self, window_widget):
        assert window_widget.ui.view_widget.sizePolicy() == QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

    def test_view_layout(self, window_widget):
        assert isinstance(window_widget.ui.view_layout, QHBoxLayout)

    def test_add_view_layout_to_widget(self, window_widget):
        assert window_widget.ui.view_widget.layout() == window_widget.ui.view_layout

    # Test results display
    def test_ui_initialises_results_display(self, window_widget):
        assert isinstance(window_widget.ui.results_display, ImageViewer)

    def test_adds_results_display_to_view_layout(self, window_widget):
        assert (
            window_widget.ui.view_layout.itemAt(0).widget()
            == window_widget.ui.results_display
        )

    # Test scan viewer
    def test_ui_initialises_scan_viewer(self, window_widget):
        assert isinstance(window_widget.ui.scan_display, ScanViewer)

    def test_ui_initialises_scan_viewer_not_visible(self, window_widget):
        assert window_widget.ui.scan_display.isVisible() is False

    def test_scan_viewer_added_to_splitter(self, window_widget):
        assert window_widget.ui.splitter.widget(3) == window_widget.ui.scan_display

    # Test scan viewer toggle
    def test_ui_initiliases_scan_viewer_toggle(self, window_widget):
        assert isinstance(window_widget.ui.scan_display_toggle, QCheckBox)

    def test_scan_viewer_toggle_added_to_control_layout(self, window_widget):
        assert (
            window_widget.ui.scan_control_layout.itemAt(0).widget()
            == window_widget.ui.scan_display_toggle
        )

    def test_scan_viewer_toggle_initialised_to_not_checked(self, window_widget):
        assert window_widget.ui.scan_display_toggle.isChecked() is False

    def test_scan_viewer_toggle_text(self, window_widget):
        assert window_widget.ui.scan_display_toggle.text() == "Toggle 3D Scan View"

    # Test update scan button
    def test_ui_initiliases_scan_update_button(self, window_widget):
        assert isinstance(window_widget.ui.update_scan_button, QPushButton)

    def test_update_scan_button_added_to_control_layout(self, window_widget):
        assert (
            window_widget.ui.scan_control_layout.itemAt(1).widget()
            == window_widget.ui.update_scan_button
        )

    def test_update_scan_button_text(self, window_widget):
        assert window_widget.ui.update_scan_button.text() == "Update 3D Scan View"

    def test_update_scan_button_initialised_to_disabled(self, window_widget):
        assert window_widget.ui.update_scan_button.isEnabled() is False

    def test_update_scan_button_initialised_to_hidden(self, window_widget):
        assert window_widget.ui.update_scan_button.isHidden() is True

    # Test warp viewer
    def test_ui_initialises_warp_viewer(self, window_widget):
        assert isinstance(window_widget.ui.warp_display, ImageViewer)

    def test_warp_viewer_not_visible_on_initialisation(self, window_widget):
        assert window_widget.ui.warp_display.isHidden() is True

    def test_warp_viewer_added_to_splitter(self, window_widget):
        assert window_widget.ui.splitter.widget(2) == window_widget.ui.warp_widget

    # Test warp viewer
    def test_ui_initialises_tissue_sectioning_viewer(self, window_widget):
        assert isinstance(window_widget.ui.tissue_sectioning_display, ImageViewer)

    def test_tissue_sectioning_viewer_not_visible_on_initialisation(
        self, window_widget
    ):
        assert window_widget.ui.tissue_sectioning_display.isHidden() is True

    def test_tissue_sectioning_viewer_added_to_splitter(self, window_widget):
        assert (
            window_widget.ui.splitter.widget(4)
            == window_widget.ui.tissue_sectioning_widget
        )


class TestViewerWidget:
    def test_data_is_empty(self, window_widget):
        assert window_widget.data == {}

    def test_blank_frames_start_as_empty(self, window_widget):
        assert window_widget.blank_frames == {}

    def test_default_orientation(self, window_widget):
        assert window_widget.orientation == SliceDirection.AXIAL

    def test_default_scan_display_flag(self, window_widget):
        assert window_widget.scan_display_enabled is False


class TestViewWidgetSetData:
    def test_set_data_sets_data(self, window_widget):
        data = {
            "scan_1": np.ones((3, 3, 3), dtype=np.float64),
            "scan_2": np.ones((3, 3, 3), dtype=np.float64) * 2,
        }
        window_widget.set_data(data)
        np.testing.assert_array_equal(
            window_widget.data["scan_1"], data["scan_1"] / data["scan_1"].max()
        )
        np.testing.assert_array_equal(
            window_widget.data["scan_2"], data["scan_2"] / data["scan_2"].max()
        )

    def test_set_data_setup_orientation(self, qtbot):
        data = {
            "scan_1": np.ones((3, 3, 3), dtype=np.float64),
            "scan_2": np.ones((3, 3, 3), dtype=np.float64) * 2,
        }
        with mock.patch.object(
            ViewerWidget, "setup_orientation"
        ) as mock_setup_orientation:
            with mock.patch.object(ViewerWidget, "display_results"):
                window = ViewerWidget(2)
                qtbot.addWidget(window)
                window.set_data(data)
                mock_setup_orientation.assert_called_with(window.orientation)


class TestViewerWidgetSetupOrientation:
    def test_setup_orientation_sets_blank_frames(self, window_widget):
        data = {
            "scan_1": np.ones((3, 3, 5), dtype=np.float64),
            "scan_2": np.ones((3, 4, 8), dtype=np.float64),
        }
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.AXIAL)
        np.testing.assert_array_equal(
            window_widget.blank_frames["scan_1"], np.zeros((3, 5))
        )
        np.testing.assert_array_equal(
            window_widget.blank_frames["scan_2"], np.zeros((4, 8))
        )

    def test_setup_orientation_keeps_minimum_slider_position(self, window_widget):
        data = {"scan_1": np.ones((3, 3, 3)), "scan_2": np.ones((3, 3, 3))}
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.AXIAL)
        assert window_widget.ui.slice_slider.minimum() == 0

    def test_setup_orientation_sets_maximum_slider(self, window_widget):
        data = {"scan_1": np.ones((50, 3, 3)), "scan_2": np.ones((50, 3, 3))}
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.AXIAL)
        assert window_widget.ui.slice_slider.maximum() == 49

    def test_setup_orientation_sets_new_orientation(self, window_widget):
        data = {"scan_1": np.ones((50, 3, 3)), "scan_2": np.ones((50, 3, 3))}
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.CORONAL)
        assert window_widget.orientation == SliceDirection.CORONAL

    # TODO: Add tests for other orientations of blank frame shapes


class TestViewerWidgetDisplaySlice:
    def test_display_slice(self, qtbot):
        data = {
            "scan_1": np.zeros((3, 3, 3), dtype=np.float64),
            "scan_2": np.zeros((3, 3, 3), dtype=np.float64) * 2,
        }
        data["scan_1"][1, :, :] = 1.0
        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.set_data(data)
            window.display_slice(1)
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[3][0][0],
                np.array(
                    [
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                        [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
                    ]
                )[0],
            )

    def test_slider_changes_image(self, qtbot):
        data = {"scan_1": np.ones((50, 3, 3)), "scan_2": np.ones((50, 3, 3))}
        with mock.patch.object(ViewerWidget, "display_slice") as mock_display_slice:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.set_data(data)
            window.ui.slice_slider.valueChanged.emit(10)
            mock_display_slice.assert_called_with(10)

    def test_display_slice_blank_offset(self, qtbot):
        data = {"scan_1": np.ones((5, 3, 3)), "scan_2": np.ones((5, 4, 4))}
        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.set_data(data)
            window.display_slice(10)
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[3][0][0], np.zeros((3, 3))
            )
            assert mock_set_image.call_args_list[3][0][1] == "Scan 1: Slice 10"
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[4][0][0], np.zeros((4, 4))
            )
            assert mock_set_image.call_args_list[4][0][1] == "Scan 2: Slice 10"

    def test_display_slice_blank_offset_different(self, qtbot):
        data = {"scan_1": np.ones((10, 3, 3)), "scan_2": np.ones((12, 4, 4))}
        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.set_data(data)
            window.display_slice(11)
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[3][0][0], np.zeros((3, 3))
            )
            assert mock_set_image.call_args_list[3][0][1] == "Scan 1: Slice 11"
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[4][0][0], np.ones((4, 4))
            )
            assert mock_set_image.call_args_list[4][0][1] == "Scan 2: Slice 11"


class TestViewerWidgetAlignmentControls:
    def test_buttons_change_results_image(self, qtbot):
        data = {"scan_1": np.ones((50, 3, 3)), "scan_2": np.ones((50, 3, 3))}
        with mock.patch.object(ViewerWidget, "display_results") as mock_display_results:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.set_data(data)
            window.ui.slice_slider.setValue(17)
            for button in window.ui.alignment_buttons:
                mock_display_results.reset_mock()
                button.toggled.emit(None)
                mock_display_results.assert_called_with(17)

    def test_phase_correlation_controls_shown_when_phase_correlation_radio_button_pressed(
        self, window_widget
    ):
        window_widget.ui.phase_correlation_button.setChecked(True)
        for widget in [
            window_widget.ui.local_region_x_label,
            window_widget.ui.local_region_x,
            window_widget.ui.local_region_y_label,
            window_widget.ui.local_region_y,
            window_widget.ui.phase_correlation_push_button,
        ]:
            assert not widget.isHidden()

    def test_phase_correlation_controls_hidden_when_other_alignement_radio_button_pressed(
        self, window_widget
    ):
        window_widget.ui.phase_correlation_button.setChecked(True)
        window_widget.ui.unaligned_button.setChecked(True)
        for widget in [
            window_widget.ui.local_region_x_label,
            window_widget.ui.local_region_x,
            window_widget.ui.local_region_y_label,
            window_widget.ui.local_region_y,
            window_widget.ui.phase_correlation_push_button,
        ]:
            assert widget.isHidden()

    @pytest.mark.parametrize("coords", ([100, 100], [111, 222], [500, 200]))
    def test_local_region_updated_on_input(self, window_widget, coords):
        data = {"scan_1": np.ones((50, 3, 3)), "scan_2": np.ones((50, 3, 3))}
        window_widget.set_data(data)
        window_widget.ui.phase_correlation_button.setChecked(True)
        window_widget.ui.local_region_x.setValue(coords[0])
        window_widget.ui.local_region_y.setValue(coords[1])
        assert window_widget.local_region == coords

    def test_local_region_set_to_none_when_phase_correlation_unchecked(
        self, window_widget
    ):
        data = {"scan_1": np.ones((50, 3, 3)), "scan_2": np.ones((50, 3, 3))}
        window_widget.set_data(data)
        window_widget.ui.phase_correlation_button.setChecked(True)
        window_widget.ui.local_region_x.setValue(200)
        window_widget.ui.local_region_y.setValue(300)
        window_widget.ui.unaligned_button.setChecked(True)
        assert window_widget.local_region is None

    def test_local_phase_correlation_started_when_button_pressed(self, window_widget):
        window_widget.ui.phase_correlation_button.setChecked(True)
        with mock.patch.object(
            ViewerWidget, "show_phase_correlation"
        ) as mock_show_phase_correlation:
            window_widget.ui.phase_correlation_push_button.click()
            mock_show_phase_correlation.assert_called()

    def test_warp_viewer_visible_when_cpd_button_pressed(self, window_widget):
        window_widget.ui.non_rigid_2d_button.setChecked(True)
        assert window_widget.ui.warp_display.isHidden() is False

    def test_warp_viewer_not_visible_when_cpd_button_unchecked(self, window_widget):
        window_widget.ui.non_rigid_2d_button.setChecked(True)
        window_widget.ui.unaligned_button.setChecked(True)
        assert window_widget.ui.warp_display.isHidden() is True


class TestViewerWidgetDisplayResults:
    @pytest.fixture()
    def mock_data(self):
        data = {
            "scan_1": np.zeros((1000, 1000, 1000), dtype=np.uint8),
            "scan_2": np.zeros((1000, 1000, 1000), dtype=np.uint8),
        }
        data["scan_1"][450:500, 500, 450:500] = 50
        data["scan_1"][500:550, 500, 450:500] = 100
        data["scan_1"][450:500, 500, 500:550] = 150
        data["scan_1"][500:550, 500, 500:550] = 200
        data["scan_2"][650:700, 500, 450:500] = 50
        data["scan_2"][700:750, 500, 450:500] = 100
        data["scan_2"][650:700, 500, 500:550] = 150
        data["scan_2"][700:750, 500, 500:550] = 200
        return data

    def test_unaligned_image_overlay_displayed(self, window_widget):
        data = {
            "scan_1": np.random.rand(30, 40, 50),
            "scan_2": np.random.rand(30, 40, 50),
        }
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.CORONAL)
        window_widget.ui.unaligned_button.setChecked(True)

        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window_widget.display_results(20)
            mock_set_image.assert_called()
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[0][0][0],
                (
                    generate_overlay_2d(
                        [data["scan_1"][:, 20, :], data["scan_2"][:, 20, :]]
                    )
                ),
            )

    def test_sift_aligned_image_overlay_displayed(self, window_widget, mock_data):
        window_widget.data = mock_data
        window_widget.setup_orientation(SliceDirection.CORONAL)

        expected = generate_overlay_2d(
            [
                mock_data["scan_1"][:, 500, :],
                align_image(
                    mock_data["scan_2"][:, 500, :],
                    mock_data["scan_1"][:, 500, :],
                    detector="SIFT",
                ),
            ]
        )

        window_widget.ui.slice_slider.setValue(500)

        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window_widget.ui.sift_button.setChecked(True)
            mock_set_image.assert_called()
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[0][0][0], expected
            )

    def test_orb_aligned_image_overlay_displayed(self, window_widget, mock_data):
        window_widget.data = mock_data
        window_widget.setup_orientation(SliceDirection.CORONAL)
        expected = generate_overlay_2d(
            [
                mock_data["scan_1"][:, 500, :],
                align_image(
                    mock_data["scan_2"][:, 500, :],
                    mock_data["scan_1"][:, 500, :],
                    detector="ORB",
                ),
            ]
        )

        window_widget.ui.slice_slider.setValue(500)

        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window_widget.ui.orb_button.setChecked(True)
            mock_set_image.assert_called()
            # TODO: for some reason the actual result doesn't exactly match the expected
            assert (
                np.count_nonzero(mock_set_image.call_args_list[0][0][0] != expected)
                < 200
            )

    @pytest.mark.parametrize(
        "scan_1_max,scan_2_max,error_string",
        (
            [45, 40, "Scan 2 index out of bounds"],
            [40, 45, "Scan 1 index out of bounds"],
        ),
    )
    def test_blank_image_shown_when_index_exceeded(
        self, window_widget, scan_1_max, scan_2_max, error_string
    ):
        data = {
            "scan_1": np.random.rand(30, scan_1_max, 50),
            "scan_2": np.random.rand(30, scan_2_max, 50),
        }
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.CORONAL)
        window_widget.ui.slice_slider.setValue(42)
        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window_widget.ui.sift_button.setChecked(True)
            mock_set_image.assert_called()
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[0][0][0], np.zeros((30, 50))
            )
            assert mock_set_image.call_args_list[0][0][1] == error_string

    def test_not_enough_keypoints_error(self, window_widget):
        data = {
            "scan_1": np.random.rand(30, 40, 50),
            "scan_2": np.random.rand(30, 40, 50),
        }
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.CORONAL)
        window_widget.ui.slice_slider.setValue(42)
        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window_widget.ui.sift_button.setChecked(True)
            mock_set_image.assert_called()
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[0][0][0], np.zeros((30, 50))
            )
            assert (
                mock_set_image.call_args_list[0][0][1]
                == "Failed to match enough keypoints"
            )

    def test_result_before_phase_correlation_shift_calculated(self, window_widget):
        data = {
            "scan_1": np.random.rand(30, 40, 50),
            "scan_2": np.random.rand(30, 40, 50),
        }
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.CORONAL)
        window_widget.ui.slice_slider.setValue(21)
        expected = generate_overlay_2d(
            [data["scan_1"][:, 21, :], data["scan_2"][:, 21, :]]
        )
        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window_widget.ui.phase_correlation_button.setChecked(True)
            mock_set_image.assert_called()
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[0][0][0], expected
            )
            assert (
                mock_set_image.call_args_list[0][0][1]
                == "Unaligned: Phase correlation shift not calculated"
            )

    def test_result_after_phase_correlation_shift_calculated(self, window_widget):
        data = {
            "scan_1": np.random.rand(30, 40, 50),
            "scan_2": np.random.rand(30, 40, 50),
        }
        shift = np.array([14, -1, 5])
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.CORONAL)
        window_widget.ui.slice_slider.setValue(21)
        window_widget.phase_correlation_shift = shift
        expected = generate_overlay_2d(
            [data["scan_1"][:, 21, :], shift_nd(data["scan_2"], -shift)[:, 21, :]]
        )
        with mock.patch.object(ImageViewer, "set_image") as mock_set_image:
            window_widget.ui.phase_correlation_button.setChecked(True)
            mock_set_image.assert_called()
            np.testing.assert_array_equal(
                mock_set_image.call_args_list[0][0][0], expected
            )
            assert mock_set_image.call_args_list[0][0][1] == "Phase correlation aligned"


class TestUpdateScanToggle:
    def test_sets_flag_depending_on_toggle(self, window_widget):
        assert window_widget.scan_display_enabled is False
        window_widget.ui.scan_display_toggle.setChecked(True)
        assert window_widget.scan_display_enabled is True
        window_widget.ui.scan_display_toggle.setChecked(False)
        assert window_widget.scan_display_enabled is False

    def test_sets_scan_display_visiblity_depending_on_toggle(self, window_widget):
        assert window_widget.ui.scan_display.isHidden() is True
        window_widget.ui.scan_display_toggle.setChecked(True)
        assert window_widget.ui.scan_display.isHidden() is False
        window_widget.ui.scan_display_toggle.setChecked(False)
        assert window_widget.ui.scan_display.isHidden() is True

    def test_update_scan_toggle_calls_update_scan_with_slider_value(
        self, window_widget
    ):
        with mock.patch.object(ViewerWidget, "update_scan") as mock_update_scan:
            window_widget.ui.scan_display_toggle.setChecked(True)
            mock_update_scan.assert_called_with(window_widget.ui.slice_slider.value())


class TestTissueSectioningCheckbox:
    @pytest.fixture()
    def window_widget_with_data(self, window_widget):
        data = {
            "scan_1": np.indices((20, 3, 3), dtype=np.float64),
            "scan_2": np.indices((20, 3, 3), dtype=np.float64),
        }
        window_widget.data = data
        window_widget.maxes = {"scan_1": 10, "scan_2": 20}
        window_widget.sectioner = None
        window_widget.blank_frames["scan_1"] = np.zeros((10, 10))
        return window_widget

    def test_tissue_sectioning_display_shown_when_checked(
        self, window_widget_with_data
    ):
        window_widget_with_data.ui.tissue_sectioning_checkbox.setChecked(True)
        assert window_widget_with_data.ui.tissue_sectioning_display.isHidden() is False

    def test_tissue_sectioning_display_shown_when_unchecked(
        self, window_widget, window_widget_with_data
    ):
        window_widget_with_data.ui.tissue_sectioning_checkbox.setChecked(True)
        window_widget_with_data.ui.tissue_sectioning_checkbox.setChecked(False)
        assert window_widget_with_data.ui.tissue_sectioning_display.isHidden() is True


class TestUpdateScan:
    def test_update_scan_calls_display_data_when_enabled(self, qtbot):
        data = {
            "scan_1": np.indices((20, 3, 3), dtype=np.float64),
            "scan_2": np.indices((20, 3, 3), dtype=np.float64),
        }
        with mock.patch.object(ScanViewer, "display_data") as mock_display_data:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.data = data
            window.scan_display_enabled = True
            window.update_scan(5)
            mock_display_data.assert_called_with(
                5, 0, slice_through_scan=False, show_navigation_slice=True
            )

    def test_update_scan_calls_display_data_when_enabled_with_orientation(self, qtbot):
        data = {
            "scan_1": np.indices((20, 3, 3), dtype=np.float64),
            "scan_2": np.indices((20, 3, 3), dtype=np.float64),
        }
        with mock.patch.object(ScanViewer, "display_data") as mock_display_data:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.data = data
            window.scan_display_enabled = True
            window.change_orientation(2)
            window.update_scan(5)
            mock_display_data.assert_called_with(
                5, 2, slice_through_scan=False, show_navigation_slice=True
            )

    def test_update_scan_does_not_call_display_data_when_disabled(self, qtbot):
        data = {
            "scan_1": np.indices((20, 3, 3), dtype=np.float64),
            "scan_2": np.indices((20, 3, 3), dtype=np.float64),
        }
        with mock.patch.object(ScanViewer, "display_data") as mock_display_data:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.data = data
            window.scan_display_enabled = False
            window.update_scan(5)
            assert mock_display_data.called is False

    def test_update_scan_button_disables_update_button(self, window_widget):
        window_widget.ui.update_scan_button.setEnabled(True)
        window_widget.update_scan(5)
        assert window_widget.ui.update_scan_button.isEnabled() is False

    def test_update_scan_button_enabled_with_slider(self, window_widget):
        data = {
            "scan_1": np.ones((20, 3, 3), dtype=np.float64),
            "scan_2": np.ones((20, 3, 3), dtype=np.float64),
        }
        window_widget.set_data(data)
        assert window_widget.ui.update_scan_button.isEnabled() is False
        window_widget.ui.slice_slider.valueChanged.emit(5)
        assert window_widget.ui.update_scan_button.isEnabled() is True


class TestViewerWidgetChangeOrientation:
    def test_change_orientation_sets_orientation(self, window_widget):
        data = {"scan_1": np.ones((50, 3, 3)), "scan_2": np.ones((50, 3, 3))}
        window_widget.data = data
        window_widget.setup_orientation(SliceDirection.CORONAL)
        assert window_widget.orientation == SliceDirection.CORONAL

    def test_change_orientation_signal_sets_orientation(self, window_widget):
        data = {"scan_1": np.ones((50, 3, 3)), "scan_2": np.ones((50, 3, 3))}
        window_widget.data = data
        window_widget.ui.orientation_selection.setCurrentIndex(1)
        assert window_widget.orientation == SliceDirection.CORONAL

    def test_change_orientation_calls_setup_orientation_with_index(self, qtbot):
        data = {
            "scan_1": np.indices((3, 3, 3), dtype=np.float64),
            "scan_2": np.indices((3, 3, 3), dtype=np.float64),
        }
        with mock.patch.object(
            ViewerWidget, "setup_orientation"
        ) as mock_setup_orientation:
            with mock.patch.object(ViewerWidget, "display_results"):
                window = ViewerWidget(2)
                qtbot.addWidget(window)
                window.data = data
                window.change_orientation(2)
                mock_setup_orientation.assert_called_with(SliceDirection(2))

    def test_change_orientation_calls_setup_orientation_with_different_index(
        self, qtbot
    ):
        data = {
            "scan_1": np.indices((3, 3, 3), dtype=np.float64),
            "scan_2": np.indices((3, 3, 3), dtype=np.float64),
        }
        with mock.patch.object(
            ViewerWidget, "setup_orientation"
        ) as mock_setup_orientation:
            with mock.patch.object(ViewerWidget, "display_results"):
                window = ViewerWidget(2)
                qtbot.addWidget(window)
                window.data = data
                window.change_orientation(1)
                mock_setup_orientation.assert_called_with(SliceDirection(1))

    def test_change_orientation_calls_display_slice_with_slider_position(self, qtbot):
        data = {
            "scan_1": np.indices((20, 3, 3), dtype=np.float64),
            "scan_2": np.indices((20, 3, 3), dtype=np.float64),
        }
        with mock.patch.object(ViewerWidget, "display_slice") as mock_display_slice:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.ui.slice_slider = mock.MagicMock()
            window.ui.slice_slider.value.return_value = 15
            window.data = data
            window.change_orientation(1)
            mock_display_slice.assert_called_with(15)

    def test_change_orientation_calls_display_results_with_slider_position(self, qtbot):
        data = {
            "scan_1": np.indices((20, 3, 3), dtype=np.float64),
            "scan_2": np.indices((20, 3, 3), dtype=np.float64),
        }
        with mock.patch.object(ViewerWidget, "display_results") as mock_display_results:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.ui.slice_slider = mock.MagicMock()
            window.ui.slice_slider.value.return_value = 15
            window.data = data
            window.change_orientation(1)
            mock_display_results.assert_called_with(15)

    def test_change_orientation_calls_update_scan(self, qtbot):
        data = {
            "scan_1": np.indices((20, 3, 3), dtype=np.float64),
            "scan_2": np.indices((20, 3, 3), dtype=np.float64),
        }
        with mock.patch.object(ViewerWidget, "update_scan") as mock_update_scan:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.data = data
            window.ui.slice_slider = mock.MagicMock()
            window.ui.slice_slider.value.return_value = 15
            window.scan_display_enabled = True
            window.change_orientation(1)
            mock_update_scan.assert_called_with(15)

    def test_change_orientation_does_not_call_update_scan_if_disabled(self, qtbot):
        data = {
            "scan_1": np.indices((20, 3, 3), dtype=np.float64),
            "scan_2": np.indices((20, 3, 3), dtype=np.float64),
        }
        with mock.patch.object(ViewerWidget, "update_scan") as mock_update_scan:
            window = ViewerWidget(2)
            qtbot.addWidget(window)
            window.data = data
            window.ui.slice_slider = mock.MagicMock()
            window.ui.slice_slider.value.return_value = 15
            window.scan_display_enabled = False
            window.change_orientation(1)
            assert mock_update_scan.called is False

    def test_change_orientation_signal_triggers_update_scan_button(self, window_widget):
        data = {
            "scan_1": np.indices((20, 3, 3), dtype=np.float64),
            "scan_2": np.indices((20, 3, 3), dtype=np.float64),
        }
        window_widget.data = data
        assert window_widget.ui.update_scan_button.isEnabled() is False
        window_widget.ui.orientation_selection.currentIndexChanged.emit(2)
        assert window_widget.ui.update_scan_button.isEnabled() is True


class TestDetectEllipses:
    @pytest.fixture()
    def window_widget_with_data(self, window_widget, sectioner_patch):
        sphere_data = np.fft.ifftshift(sphere([50, 50, 50], radius=20))

        data = {
            "scan_1": sphere_data.astype(np.float64),
            "scan_2": sphere_data.astype(np.float64),
        }
        with mock.patch.object(ScanViewer, "set_data"):
            window_widget.set_data(data)
            window_widget.maxes = {"scan_1": 10, "scan_2": 20}
            window_widget.sectioner = sectioner_patch
            window_widget.blank_frames["scan_1"] = np.zeros((10, 10))
        return window_widget

    def test_detect_ellipses(self, window_widget_with_data):
        with mock.patch.object(ViewerWidget, "_display_info"):
            window_widget_with_data.detect_ellipses()


class TestUpdateEllipseTable:
    def test_adds_item_to_table(self, window_widget):
        ellipses = [{"centre": None, "volume": 10, "class": 3}]
        window_widget.update_ellipse_table(ellipses, [])
        assert window_widget.ui.detection_tables[0].rowCount() == 1
        assert window_widget.ui.detection_tables[1].rowCount() == 0

    def test_adds_item_to_second_table(self, window_widget):
        ellipses = [{"centre": None, "volume": 10, "class": 3}]
        window_widget.update_ellipse_table([], ellipses)
        assert window_widget.ui.detection_tables[0].rowCount() == 0
        assert window_widget.ui.detection_tables[1].rowCount() == 1

    def test_set_table_1_headers(self, window_widget):
        window_widget.update_ellipse_table([], [])
        assert (
            window_widget.ui.detection_tables[0].horizontalHeaderItem(0).text()
            == "Lesion ID"
        )
        assert (
            window_widget.ui.detection_tables[0].horizontalHeaderItem(1).text()
            == "Centre"
        )
        assert (
            window_widget.ui.detection_tables[0].horizontalHeaderItem(2).text()
            == "Volume"
        )
        assert (
            window_widget.ui.detection_tables[0].horizontalHeaderItem(3).text()
            == "Class"
        )

    def test_set_table_2_headers(self, window_widget):
        window_widget.update_ellipse_table([], [])
        assert (
            window_widget.ui.detection_tables[1].horizontalHeaderItem(0).text()
            == "Lesion ID"
        )
        assert (
            window_widget.ui.detection_tables[1].horizontalHeaderItem(1).text()
            == "Centre"
        )
        assert (
            window_widget.ui.detection_tables[1].horizontalHeaderItem(2).text()
            == "Volume"
        )
        assert (
            window_widget.ui.detection_tables[1].horizontalHeaderItem(3).text()
            == "Class"
        )

    def test_sets_item_information(self, window_widget):
        ellipses = [{"centre": [1, 2, 3], "volume": 10, "class": 3}]
        window_widget.update_ellipse_table(ellipses, [])
        assert window_widget.ui.detection_tables[0].item(0, 0).text() == "0"
        assert (
            window_widget.ui.detection_tables[0].item(0, 1).text() == "[1.00 2.00 3.00]"
        )
        assert window_widget.ui.detection_tables[0].item(0, 2).text() == "10.00"
        assert window_widget.ui.detection_tables[0].item(0, 3).text() == "3"

    def test_sets_item_information_table_2(self, window_widget):
        ellipses = [{"centre": [1, 2, 3], "volume": 10, "class": 3}]
        window_widget.update_ellipse_table([], ellipses)
        assert window_widget.ui.detection_tables[1].item(0, 0).text() == "0"
        assert (
            window_widget.ui.detection_tables[1].item(0, 1).text() == "[1.00 2.00 3.00]"
        )
        assert window_widget.ui.detection_tables[1].item(0, 2).text() == "10.00"
        assert window_widget.ui.detection_tables[1].item(0, 3).text() == "3"

    def test_detection_table_hidden_by_default(self, window_widget):
        assert window_widget.ui.detection_widget.isHidden() is True

    def test_update_ellipse_table_unhides_table(self, window_widget):
        window_widget.update_ellipse_table([], [])
        assert window_widget.ui.detection_widget.isHidden() is False


class TestHandleItemSelection:
    @pytest.fixture()
    def window_widget_with_data(self, window_widget, sectioner_patch):
        sphere_data = np.fft.ifftshift(sphere([50, 50, 50], radius=20))

        data = {
            "scan_1": sphere_data.astype(np.float64),
            "scan_2": sphere_data.astype(np.float64),
        }
        with mock.patch.object(ScanViewer, "set_data"):
            window_widget.set_data(data)
            window_widget.maxes = {"scan_1": 10, "scan_2": 20}
            window_widget.sectioner = sectioner_patch
            window_widget.blank_frames["scan_1"] = np.zeros((10, 10))
        return window_widget

    def test_handle_item_selection_1(self, window_widget_with_data):
        ellipses = [
            {
                "centre": [10.1, 2, 3],
                "volume": 10,
                "class": 3,
                "contributing_ellipses": [],
            }
        ]
        window_widget_with_data.ellipses = {"scan_1": ellipses, "scan_2": []}
        window_widget_with_data.update_ellipse_table(ellipses, [])
        window_widget_with_data.ui.detection_tables[0].selectRow(0)
        assert window_widget_with_data.ui.slice_slider.value() == 10

    def test_handle_item_selection_2(self, window_widget_with_data):
        ellipses = [
            {
                "centre": [10.1, 2, 3],
                "volume": 10,
                "class": 3,
                "contributing_ellipses": [],
            }
        ]
        window_widget_with_data.ellipses = {"scan_1": [], "scan_2": ellipses}
        window_widget_with_data.update_ellipse_table([], ellipses)
        window_widget_with_data.ui.detection_tables[1].selectRow(0)
        assert window_widget_with_data.ui.slice_slider.value() == 10

    def test_handle_item_selection_2_whitespace(self, window_widget_with_data):
        ellipses = [
            {
                "centre": [10.1, 2, 3],
                "volume": 10,
                "class": 3,
                "contributing_ellipses": [],
            }
        ]
        window_widget_with_data.ellipses = {"scan_1": [], "scan_2": ellipses}
        window_widget_with_data.update_ellipse_table([], ellipses)
        window_widget_with_data.ui.detection_tables[1].selectRow(0)
        assert window_widget_with_data.ui.slice_slider.value() == 10
