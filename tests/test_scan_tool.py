import mock
import sys
import pytest
from PySide2.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction
from PySide2.QtCore import QSize

from ai_ct_scans import scan_tool


class TestScanTool:
    def test_calls_app_exec(self):
        with mock.patch.object(QMainWindow, "show"):
            with mock.patch.object(sys, "exit"):
                with mock.patch.object(QApplication, "exec_") as mock_app_exec:
                    scan_tool.main()
        assert mock_app_exec.call_count == 1


class TestMainWindow:
    @pytest.fixture
    def window_widget(self, qtbot):
        window = scan_tool.MainWindow()
        qtbot.addWidget(window)
        return window

    def test_main_window_is_qmainwindow_widget(self):
        assert issubclass(scan_tool.MainWindow, QMainWindow)

    def test_set_window_title(self, window_widget):
        assert window_widget.windowTitle() == "AI CT Scan Tool"

    def test_resizes_window_to_set_size(self, window_widget):
        assert window_widget.size() == QSize(1024, 768)

    def test_stores_stack_window_in_main_window_ui(self, window_widget):
        assert isinstance(window_widget.ui.central_widget, scan_tool.UiStackWindow)

    def test_sets_central_widget_to_UiStackWindow_instance(self, window_widget):
        assert window_widget.centralWidget() == window_widget.ui.central_widget

    def test_set_menubar(self, window_widget):
        assert isinstance(window_widget.ui.menu_bar, QMenuBar)

    def test_set_file_menu(self, window_widget):
        assert isinstance(window_widget.ui.file_menu, QMenu)

    def test_set_load_action(self, window_widget):
        assert isinstance(window_widget.ui.load_action, QAction)

    def test_load_action_text(self, window_widget):
        assert window_widget.ui.load_action.text() == "&Load Data"

    def test_set_quit_action(self, window_widget):
        assert isinstance(window_widget.ui.quit_action, QAction)

    def test_quit_action_text(self, window_widget):
        assert window_widget.ui.quit_action.text() == "&Quit"

    def test_load_action_added_to_file_menu(self, window_widget):
        assert window_widget.ui.file_menu.actions()[0] == window_widget.ui.load_action

    def test_quit_action_added_to_file_menu(self, window_widget):
        assert window_widget.ui.file_menu.actions()[1] == window_widget.ui.quit_action
