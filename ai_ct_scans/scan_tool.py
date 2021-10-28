import sys

from PySide2.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction
from PySide2 import QtCore

from ai_ct_scans.GUI import UiStackWindow


class MainWindow(QMainWindow):
    """Main application window.

    Attributes:
        ui (MainWindow.Ui): UI class containing UI elements for this widget.
    """

    class Ui:
        def __init__(self, widget):
            """UI Class to contain UI elements for the MainWindow widget class.

            Args:
                widget (QWidget): parent widget to apply UI elements to.
            """
            widget.setWindowTitle("AI CT Scan Tool")
            widget.resize(1024, 768)

            self.central_widget = UiStackWindow()
            widget.setCentralWidget(self.central_widget)

            # Configure menubar
            self.menu_bar = QMenuBar(widget)
            widget.setMenuBar(self.menu_bar)

            # Add file menu (for loading data and exiting)
            self.file_menu = QMenu("&File", widget)
            self.menu_bar.addMenu(self.file_menu)

            self.load_action = QAction("&Load Data", widget)
            self.file_menu.addAction(self.load_action)

            self.quit_action = QAction("&Quit", widget)
            self.file_menu.addAction(self.quit_action)

    def __init__(self):
        super().__init__()
        self.ui = self.Ui(self)

        self.ui.load_action.triggered.connect(self.ui.central_widget.choose_data_dir)
        self.ui.quit_action.triggered.connect(self.close)


def main():
    """Main GUI application endpoint."""
    QApplication.setAttribute(QtCore.Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(QtCore.Qt.AA_UseHighDpiPixmaps, True)

    app = QApplication([])

    app.setStyle("Fusion")

    main_window = MainWindow()

    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
