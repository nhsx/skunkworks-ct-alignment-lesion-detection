from pathlib import Path
import mock

import pytest
from PySide2.QtWidgets import QSizePolicy, QFileDialog, QMdiArea

from ai_ct_scans.GUI import UiStackWindow, ViewerWidget
from ai_ct_scans.GUI import ui_stack_window


@pytest.fixture
def window_widget(qtbot):
    window = UiStackWindow()
    qtbot.addWidget(window)
    return window


class TestUiStackWindow:
    def test_ui_stack_window_is_mdi_area(self):
        assert issubclass(UiStackWindow, QMdiArea)

    def test_ui_size_policy(self, window_widget):
        assert window_widget.sizePolicy() == QSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

    def test_choose_data_directory_triggers_dialog(self, qtbot):
        with mock.patch.object(
            QFileDialog, "getExistingDirectory"
        ) as mock_get_directory:
            with mock.patch.object(UiStackWindow, "data_path"):
                window = UiStackWindow()
                qtbot.addWidget(window)
                mock_get_directory.return_value = "path/to/directory"
                window.choose_data_dir()
                mock_get_directory.assert_called_with(
                    window,
                    "Select Data Directory",
                    options=QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
                )

    def test_choose_data_directory_emits_signal(self, qtbot):
        with mock.patch.object(
            QFileDialog, "getExistingDirectory"
        ) as mock_get_directory:
            with mock.patch.object(UiStackWindow, "data_path") as mock_data_signal:
                window = UiStackWindow()
                qtbot.addWidget(window)
                mock_get_directory.return_value = "path/to/directory"
                window.choose_data_dir()
                mock_data_signal.emit.assert_called_with(Path("path/to/directory"))

    def test_calls_load_data_from_signal(self, qtbot):
        with mock.patch.object(UiStackWindow, "load_data") as mock_load_data:
            window = UiStackWindow()
            qtbot.addWidget(window)
            window.data_path.emit(Path("path/to/directory"))
            mock_load_data.assert_called_with(Path("path/to/directory"))

    def test_set_to_tab_view(self, qtbot):
        with mock.patch.object(UiStackWindow, "load_data"):
            window = UiStackWindow()
            qtbot.addWidget(window)
            window.data_path.emit(Path("path/to/directory"))
            assert window.viewMode() == QMdiArea.TabbedView

    def test_set_tabs_moveable(self, qtbot):
        with mock.patch.object(UiStackWindow, "load_data"):
            window = UiStackWindow()
            qtbot.addWidget(window)
            window.data_path.emit(Path("path/to/directory"))
            assert window.tabsMovable() is True

    def test_set_tabs_closable(self, qtbot):
        with mock.patch.object(UiStackWindow, "load_data"):
            window = UiStackWindow()
            qtbot.addWidget(window)
            window.data_path.emit(Path("path/to/directory"))
            assert window.tabsClosable() is True


class TestUiStackWindowLoadData:
    def test_load_data_initialises_patient_loader(self, qtbot):
        with mock.patch.object(ui_stack_window, "PatientLoader") as mock_patient_loader:
            with mock.patch.object(ViewerWidget, "set_data"):
                with mock.patch.object(ui_stack_window.UiStackWindow, "_display_error"):
                    window = UiStackWindow()
                    qtbot.addWidget(window)
                    window.data_path.emit(Path("path/to/directory"))
                    mock_patient_loader.assert_called_with(
                        Path("path/to/directory"), rescale_on_load=True
                    )

    def test_error_dialog_displayed_when_scan_loading_fails(self, qtbot):
        with mock.patch.object(ui_stack_window, "PatientLoader") as mock_patient_loader:
            with mock.patch.object(
                ui_stack_window.UiStackWindow, "_display_error"
            ) as mock_display_message:
                mock_patient_loader.side_effect = FileNotFoundError(
                    "problem loading path/to/directory"
                )
                window = UiStackWindow()
                qtbot.addWidget(window)
                window.data_path.emit(Path("path/to/directory"))
                mock_display_message.assert_called_with(
                    "problem loading path/to/directory", "Scan data error"
                )

    def test_error_dialog_displayed_if_patient_transform_doesnt_exist(self, qtbot):
        with mock.patch.object(ui_stack_window, "PatientLoader"):
            with mock.patch.object(ViewerWidget, "set_data"):
                with mock.patch.object(
                    ui_stack_window.UiStackWindow, "_display_error"
                ) as mock_display_message:
                    with mock.patch.object(ui_stack_window, "sectioning"):
                        window = UiStackWindow()
                        qtbot.addWidget(window)
                        window.data_path.emit(Path("path/to/directory"))
                        mock_display_message.assert_called_with(
                            "Transform not found for patient, non-rigid alignment "
                            "will not be available.",
                            "Missing alignment transform error",
                        )

    def test_error_dialog_displayed_if_sectioning_model_doesnt_exist(self, qtbot):
        with mock.patch.object(ui_stack_window, "PatientLoader"):
            with mock.patch.object(ViewerWidget, "set_data"):
                with mock.patch.object(
                    ui_stack_window.UiStackWindow, "_display_error"
                ) as mock_display_message:
                    window = UiStackWindow()
                    qtbot.addWidget(window)
                    window.data_path.emit(Path("path/to/directory"))
                    mock_display_message.assert_called_with(
                        "Sectioning model not found, tissue sectioning will not be "
                        "available.",
                        "Missing tissue sectioning model error",
                    )
