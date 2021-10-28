import mock
import numpy as np
import pytest
import pyqtgraph as pg
import pyqtgraph.opengl as gl
from PySide2.QtWidgets import QVBoxLayout, QWidget, QSizePolicy

from ai_ct_scans.GUI import ScanViewer
from ai_ct_scans.GUI.scan_viewer import SCAN_COLOURS


class TestScanColours:
    def test_is_list(self):
        assert isinstance(SCAN_COLOURS, list)

    def test_items_are_right_size(self):
        for colour in SCAN_COLOURS:
            assert len(colour) == 4


@pytest.fixture
def viewer_widget(qtbot):
    window = ScanViewer()
    qtbot.addWidget(window)
    return window


class TestScanViewerUI:
    def test_viewer_widget_is_qwidget(self):
        assert issubclass(ScanViewer, QWidget)

    def test_ui_size_policy(self, viewer_widget):
        assert viewer_widget.sizePolicy() == QSizePolicy(
            QSizePolicy.MinimumExpanding, QSizePolicy.MinimumExpanding
        )

    def test_set_layout(self, viewer_widget):
        assert isinstance(viewer_widget.ui.layout, QVBoxLayout)

    def test_initialise_gl_viewer_widget(self, viewer_widget):
        assert isinstance(viewer_widget.ui.viewer, gl.GLViewWidget)

    def test_gl_viewer_widget_added_to_layout(self, viewer_widget):
        assert viewer_widget.ui.layout.itemAt(0).widget() == viewer_widget.ui.viewer


class TestScanViewer:
    def test_volume_plot_initialised_to_none(self, viewer_widget):
        assert viewer_widget.volume_plot is None

    def test_viewer_data_initialised_to_none(self, viewer_widget):
        assert viewer_widget.viewer_data is None


class TestSetData:
    def test_set_data_initialises_volume_plot(self, viewer_widget):
        data = {
            "scan_1": np.ones((2, 3, 4), dtype=np.float64),
            "scan_2": np.ones((2, 3, 4), dtype=np.float64) * 2,
        }
        assert viewer_widget.volume_plot is None
        viewer_widget.set_data(data)
        assert isinstance(viewer_widget.volume_plot, pg.opengl.GLVolumeItem)

    def test_centre_plot_origin(self, viewer_widget):
        # Unknown if this can be checked natively without patching.
        with mock.patch.object(pg.opengl.GLVolumeItem, "translate") as mock_translate:
            data = {
                "scan_1": np.ones((2, 3, 4), dtype=np.float64),
                "scan_2": np.ones((2, 3, 4), dtype=np.float64) * 2,
            }
            viewer_widget.set_data(data)
            mock_translate.assert_called_with(-1, -1.5, -2)

    def test_centre_plot_origin_different(self, viewer_widget):
        # Unknown if this can be checked natively without patching.
        with mock.patch.object(pg.opengl.GLVolumeItem, "translate") as mock_translate:
            data = {
                "scan_1": np.ones((2, 6, 10), dtype=np.float64),
                "scan_2": np.ones((2, 6, 10), dtype=np.float64) * 2,
            }
            viewer_widget.set_data(data)
            mock_translate.assert_called_with(-1, -3, -5)

    @pytest.mark.skip("GL errors unresolved for itemsAt for region on viewer widget.")
    def test_adds_item_to_volume_plot(self, viewer_widget):
        data = {"scan_1": np.ones((2, 3, 4), dtype=np.float64)}
        viewer_widget.set_data(data)
        assert viewer_widget.ui.viewer.itemsAt(region=(0, 0, 1, 1)) == [
            viewer_widget.volume_plot
        ]


class TestPreProcessData:
    def test_returns_numpy_array(self):
        data = {"scan_1": np.ones((2, 3, 4), dtype=np.float64)}
        assert isinstance(ScanViewer.pre_process_data(data), np.ndarray)

    def test_returns_correct_dimensions(self):
        data = {"scan_1": np.ones((2, 3, 4), dtype=np.float64)}
        # Shape should be x, y, z, RGBA
        assert ScanViewer.pre_process_data(data).shape == (2, 3, 4, 4)


class TestDisplayData:
    def test_sets_plot_data(self, viewer_widget):
        # Unknown if this can be checked natively without patching.
        with mock.patch.object(pg.opengl.GLVolumeItem, "setData") as mock_set_data:
            data = {"scan_1": np.zeros((1, 2, 2), dtype=np.float64)}
            viewer_widget.set_data(data)
            viewer_widget.display_data(
                slice_through_scan=False, show_navigation_slice=False
            )
            print(mock_set_data.call_args_list[1][0][0])
            np.testing.assert_array_equal(
                mock_set_data.call_args_list[1][0][0], viewer_widget.viewer_data
            )

    @pytest.mark.parametrize(
        "slice, expected",
        [
            (0, [[[255, 255, 255, 64]]]),
            (5, [[[255, 255, 255, 64]]]),
            (10, [[[255, 255, 255, 64]]]),
        ],
        ids=["test slice 0", "test slice 5", "test_slice 10"],
    )
    def test_sets_plot_data_with_plane(self, viewer_widget, slice, expected):
        mock_set_data = mock.MagicMock()
        viewer_widget.volume_plot = mock.MagicMock()
        viewer_widget.volume_plot.setData = mock_set_data
        viewer_widget.viewer_data = np.zeros(((15, 1, 1, 4)), dtype=np.float64)
        viewer_widget.display_data(
            slice, slice_through_scan=False, show_navigation_slice=True
        )
        np.testing.assert_array_equal(
            mock_set_data.call_args_list[0][0][0][slice, :, :, :], expected
        )
        for other_slice in range(viewer_widget.viewer_data.shape[0]):
            if other_slice != slice:
                np.testing.assert_array_equal(
                    mock_set_data.call_args_list[0][0][0][other_slice, :, :, :],
                    [[[0, 0, 0, 0]]],
                )

    @pytest.mark.parametrize(
        "slice, expected",
        [
            (0, [[[255, 255, 255, 64]]]),
            (5, [[[255, 255, 255, 64]]]),
            (10, [[[255, 255, 255, 64]]]),
        ],
        ids=["test slice 0", "test slice 5", "test_slice 10"],
    )
    def test_sets_plot_data_with_plane_coronal(self, viewer_widget, slice, expected):
        mock_set_data = mock.MagicMock()
        viewer_widget.volume_plot = mock.MagicMock()
        viewer_widget.volume_plot.setData = mock_set_data
        viewer_widget.viewer_data = np.zeros(((1, 15, 1, 4)), dtype=np.float64)
        viewer_widget.display_data(
            slice, orientation=1, slice_through_scan=False, show_navigation_slice=True
        )
        np.testing.assert_array_equal(
            mock_set_data.call_args_list[0][0][0][:, slice, :, :], expected
        )
        for other_slice in range(viewer_widget.viewer_data.shape[1]):
            if other_slice != slice:
                np.testing.assert_array_equal(
                    mock_set_data.call_args_list[0][0][0][:, other_slice, :, :],
                    [[[0, 0, 0, 0]]],
                )

    @pytest.mark.parametrize(
        "slice, expected",
        [
            (0, [[[255, 255, 255, 64]]]),
            (5, [[[255, 255, 255, 64]]]),
            (10, [[[255, 255, 255, 64]]]),
        ],
        ids=["test slice 0", "test slice 5", "test_slice 10"],
    )
    def test_sets_plot_data_with_plane_saggital(self, viewer_widget, slice, expected):
        mock_set_data = mock.MagicMock()
        viewer_widget.volume_plot = mock.MagicMock()
        viewer_widget.volume_plot.setData = mock_set_data
        viewer_widget.viewer_data = np.zeros(((1, 1, 15, 4)), dtype=np.float64)
        viewer_widget.display_data(
            slice, orientation=2, slice_through_scan=False, show_navigation_slice=True
        )
        np.testing.assert_array_equal(
            mock_set_data.call_args_list[0][0][0][:, :, slice, :], expected
        )
        for other_slice in range(viewer_widget.viewer_data.shape[2]):
            if other_slice != slice:
                np.testing.assert_array_equal(
                    mock_set_data.call_args_list[0][0][0][:, :, other_slice, :],
                    [[[0, 0, 0, 0]]],
                )

    @pytest.mark.parametrize(
        "slice", [(0), (5), (10)], ids=["test slice 0", "test slice 5", "test_slice 10"]
    )
    def test_slice_through_volume(self, viewer_widget, slice):
        with mock.patch.object(pg.opengl.GLVolumeItem, "setData") as mock_set_data:
            data = {"scan_1": np.zeros((15, 1, 1), dtype=np.float64)}
            viewer_widget.set_data(data)
            viewer_widget.display_data(
                slice, slice_through_scan=True, show_navigation_slice=False
            )
            assert mock_set_data.call_args_list[1][0][0].shape[0] == 15 - slice

    @pytest.mark.parametrize(
        "slice", [(0), (5), (10)], ids=["test slice 0", "test slice 5", "test_slice 10"]
    )
    def test_slice_through_volume_coronal(self, viewer_widget, slice):
        with mock.patch.object(pg.opengl.GLVolumeItem, "setData") as mock_set_data:
            data = {"scan_1": np.zeros((1, 15, 1), dtype=np.float64)}
            viewer_widget.set_data(data)
            viewer_widget.display_data(
                slice,
                orientation=1,
                slice_through_scan=True,
                show_navigation_slice=False,
            )
            assert mock_set_data.call_args_list[1][0][0].shape[1] == 15 - slice

    @pytest.mark.parametrize(
        "slice", [(0), (5), (10)], ids=["test slice 0", "test slice 5", "test_slice 10"]
    )
    def test_slice_through_volume_saggital(self, viewer_widget, slice):
        with mock.patch.object(pg.opengl.GLVolumeItem, "setData") as mock_set_data:
            data = {"scan_1": np.zeros((1, 1, 15), dtype=np.float64)}
            viewer_widget.set_data(data)
            viewer_widget.display_data(
                slice,
                orientation=2,
                slice_through_scan=True,
                show_navigation_slice=False,
            )
            assert mock_set_data.call_args_list[1][0][0].shape[2] == 15 - slice

    def test_does_not_set_plot_data(self, viewer_widget):
        # Unknown if this can be checked natively without patching.
        with mock.patch.object(pg.opengl.GLVolumeItem, "setData") as mock_set_data:
            viewer_widget.viewer_data = None
            viewer_widget.display_data()
            assert mock_set_data.called is False


class TestConvertForViewer:
    def test_returns_numpy_array(self):
        data = np.ones((2, 3, 4), dtype=np.float64)
        assert isinstance(ScanViewer.convert_for_viewer(data), np.ndarray)

    def test_returns_correct_dimensions(self):
        data = np.ones((2, 3, 4), dtype=np.float64)
        # Shape should be x, y, z, RGBA
        assert ScanViewer.convert_for_viewer(data).shape == (2, 3, 4, 4)
