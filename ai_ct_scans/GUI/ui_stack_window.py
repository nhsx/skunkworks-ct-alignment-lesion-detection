from pathlib import Path
import pathlib

from PySide2.QtGui import QIcon
from PySide2.QtWidgets import (
    QMdiSubWindow,
    QSizePolicy,
    QFileDialog,
    QMdiArea,
    QErrorMessage,
)
from PySide2.QtCore import Slot, Signal, Qt

from .viewer_widget import ViewerWidget
from ai_ct_scans.data_loading import PatientLoader
from ai_ct_scans.non_rigid_alignment import read_transform
from ai_ct_scans import sectioning


class UiStackWindow(QMdiArea):
    """Internal application widget for switching between views and pages within the tool.

    Attributes:
        ui (UiStackWindow.Ui): UI class containing UI elements for this widget.
    """

    class Ui:
        def __init__(self, widget):
            """UI Class to contain UI elements for the UIStackWindow widget class.

            Args:
                widget (QWidget): parent widget to apply UI elements to.
            """
            widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
            widget.setViewMode(QMdiArea.TabbedView)
            widget.setTabsMovable(True)
            widget.setTabsClosable(True)

    data_path = Signal(object)

    def __init__(self):
        super().__init__()
        self.ui = self.Ui(self)

        self.error_dialog = QErrorMessage(self)
        self.error_dialog.setWindowModality(Qt.WindowModal)

        self.data_path.connect(lambda x: self.load_data(x))

    # TODO: Test data loading.
    @Slot()
    def load_data(self, data_path):
        """Load data and pass to new instance of a viewer widget subwindow within the QMdiArea.
            If CPD alignment data is present in data directory - then load and pass to the viewer.

        Args:
            data_path (pathlib Path): path to patient data directory.
        """
        try:
            data = PatientLoader(data_path, rescale_on_load=True)
        except FileNotFoundError as e:
            self._display_error(str(e), "Scan data error")
            return
        data.abdo.scan_1.load_scan()
        data.abdo.scan_2.load_scan()

        patient_id = pathlib.PurePath(data_path).name
        alignment_path = Path(data_path) / f"patient_{patient_id}.pkl"

        cpd_alignment = None
        if alignment_path.exists():
            cpd_alignment = read_transform(alignment_path)
        else:
            self._display_error(
                "Transform not found for patient, non-rigid alignment will not be available.",
                "Missing alignment transform error",
            )

        # Set up sectioner
        model_path = (
            data_path.parent.parent
            / "hierarchical_mean_shift_tissue_sectioner_model.pkl"
        )
        sectioner = sectioning.TextonSectioner()
        try:
            sectioner.load(model_path)
        except FileNotFoundError:
            self._display_error(
                "Sectioning model not found, tissue sectioning will not be available.",
                "Missing tissue sectioning model error",
            )

        # Create new window
        subwindow = QMdiSubWindow()
        subwindow.setAttribute(
            Qt.WA_DeleteOnClose, True
        )  # Ensures tab is removed when close button pushed.
        subwindow.setWindowIcon(QIcon())
        if data.abdo.scan_1.patient_name is not None:
            subwindow.setWindowTitle(str(data.abdo.scan_1.patient_name))

        # Create new view widget
        viewer = ViewerWidget(2)
        # Assign data to new viewer window
        viewer.set_data(
            {
                "scan_1": data.abdo.scan_1.full_scan,
                "scan_2": data.abdo.scan_2.full_scan,
            },
            cpd_alignment=cpd_alignment,
            sectioning_model=sectioner,
        )
        subwindow.setWidget(viewer)

        self.addSubWindow(subwindow)
        subwindow.show()

    @Slot()
    def choose_data_dir(self):
        """Open file dialog for selecting data directory."""
        data_path = QFileDialog.getExistingDirectory(
            self,
            "Select Data Directory",
            options=QFileDialog.ShowDirsOnly | QFileDialog.DontUseNativeDialog,
        )
        if data_path:
            self.data_path.emit(Path(data_path))

    def _display_error(self, message, error_type):
        """Display an error message in a dialog.

        Args:
            message (str): The message to include in the dialog.
            error_type (str): The error type, used to control which messages are displayed. Also used for the dialog
                window title.
        """
        self.error_dialog.setWindowTitle(error_type)
        self.error_dialog.showMessage(message, error_type)
