import numpy as np
import cv2

from PySide2.QtWidgets import QLabel, QSizePolicy
from PySide2.QtGui import QImage, QPixmap, QPainter, QColor, QFont

from ai_ct_scans.image_processing_utils import normalise

# TODO: Test this class properly
class ImageViewer(QLabel):
    """Class to utilise the QLabel pixmap for displaying images in the GUI.

    Attributes:
        pixmap (QPixmap): Current value of the displayed pixmap.
    """

    def __init__(self):
        super().__init__()
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.pixmap = QPixmap()
        self.info_text = None

    def set_image(self, input_image, info_text=None, point=None, ellipse=None):
        """Convert the image for displaying within the ImageViewer as a QPixmap.

        Args:
            input_image (np.array): image to display.
            info_text (str): information text string to display on corner of image.
            point (list): coordinates of a point to overlay on the image as [x, y], or None.
            ellipse (dict): dictionary describing ellipse to overlay on image.
        """
        image = np.array(input_image)
        if image.dtype != np.uint8 and not np.any((image < 0) | (image > 1)):
            image *= 255
            image = np.uint8(image)
        else:
            image = normalise(image)

        # Make sure image has three channels (covert to BGR if nessessary)
        if image.ndim == 3:
            image = image
        else:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

        # Copy the image
        image = np.copy(image)

        if point:
            image = cv2.circle(image, point, radius=5, color=(0, 0, 255), thickness=-1)

        if ellipse is not None:
            # Ensure accurate integer arguments to cv2.ellipse, use shift to maintain
            # accuracy of non-integer inputs.
            center = (
                int(round(ellipse["center"][0] * 2 ** 10)),
                int(round(ellipse["center"][1] * 2 ** 10)),
            )
            axes = (
                int(round((ellipse["axis"][0] / 2) * 2 ** 10)),
                int(round((ellipse["axis"][1] / 2) * 2 ** 10)),
            )
            image = cv2.ellipse(
                image,
                center,
                axes,
                int(round(ellipse["angle"])),
                0,
                360,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
                10,
            )

        # Create QImage from image data
        image = (
            QImage(
                image.data,
                image.shape[1],
                image.shape[0],
                image.shape[1] * 3,
                QImage.Format_RGB888,
            )
            .rgbSwapped()
            .rgbSwapped()
        )

        # Create QPixmap from QImage
        self.pixmap = QPixmap.fromImage(image)
        self.info_text = info_text

        self.update()

    def scale_factor(self):
        """Find a suitable scale factor.

        Returns:
            (float): scale factor to fit image to window.
        """
        # Check if there is an empty pixmap
        if self.pixmap.isNull():
            return 1
        else:
            # Determine scaling to fit the window without overlap
            return min(
                self.width() / self.pixmap.width(), self.height() / self.pixmap.height()
            )

    def painter_element(self):
        """Create painter instance for rendering image.

        Returns:
            (QPainter): QPainter instance with correect scaling and rendering options.
        """
        # Create painter instance
        painter_instance = QPainter(self)

        # Determine the scale factor
        painter_instance.scale(self.scale_factor(), self.scale_factor())

        # Set rendering setttings for better rendering
        painter_instance.setRenderHint(QPainter.SmoothPixmapTransform)
        painter_instance.setRenderHint(QPainter.Antialiasing)
        return painter_instance

    def paintEvent(self, event):
        """Paint the existing pixmap when the window changes.

        Args:
            event (QEvent): automatically populated Qt event information.
        """
        # Get an instance of the painter
        painter = self.painter_element()

        # Draw pixmap
        painter.drawPixmap(0, 0, self.pixmap)

        # Write info text
        painter.setPen(QColor("white"))
        painter.setFont(QFont("Arial", 15))
        if self.info_text is not None:
            # TODO: Consider moving this to another overlay method so that it has a fixed font size with any image
            painter.drawText(10, 20, self.info_text)

        # Finish drawing
        painter.end()
