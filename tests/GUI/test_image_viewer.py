import pytest
from PySide2.QtWidgets import QLabel, QSizePolicy
from PySide2.QtGui import QPainter

from ai_ct_scans.GUI import ImageViewer


class TestImageViewer:
    @pytest.fixture
    def viewer_widget(self, qtbot):
        window = ImageViewer()
        qtbot.addWidget(window)
        return window

    def test_is_subclass_of_qlabel(self):
        assert issubclass(ImageViewer, QLabel)

    def test_ui_size_policy(self, viewer_widget):
        assert viewer_widget.sizePolicy() == QSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Expanding
        )

    def test_initialised_empty_pixmap(self, viewer_widget):
        assert viewer_widget.pixmap.isNull() is True

    def test_default_scale_factor_for_no_pixmap(self, viewer_widget):
        assert viewer_widget.scale_factor() == 1

    def test_painter_element_returns_qpainter(self, viewer_widget):
        assert isinstance(viewer_widget.painter_element(), QPainter)
