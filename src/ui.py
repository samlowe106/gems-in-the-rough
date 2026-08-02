import logging
import os
import shutil
from collections.abc import Callable

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


class Button(QPushButton):
    def __init__(self, key: str, on_click: Callable, qhbox: QHBoxLayout):
        super().__init__()

        # Named `key`, not `shortcut`: QAbstractButton already exposes a
        # `shortcut` property, and shadowing it breaks Qt's own accessor.
        self.key = key

        self._on_click_template = on_click
        self._on_click: Callable | None = None

        self.setText("Thinking...")

        qhbox.addWidget(self)

    def _disconnect_button(self):
        if self._on_click is not None:
            self.clicked.disconnect(self._on_click)
            self._on_click = None

    def set_thinking(self):
        self.setText("Thinking...")
        self._disconnect_button()
        self.setShortcut("")
        self.setEnabled(False)

    def set_unavailable(self, text: str = "Unavailable"):
        """Disable the button when there is no suggestion to act on."""
        self.setText(text)
        self._disconnect_button()
        self.setShortcut("")
        self.setEnabled(False)

    def show_result(self, result_label: str, destination_dir: str | None = None):
        if destination_dir is None:
            destination_dir = result_label
        self.setText(f"{result_label} ({self.key})")
        self._disconnect_button()
        self.setShortcut(self.key)
        self.setEnabled(True)
        self._on_click = lambda: self._on_click_template(
            destination_dir=destination_dir
        )
        self.clicked.connect(self._on_click)


class ImageSorterUI(QWidget):
    def __init__(
        self,
        manager,
        num_buttons: int,
        input_dir: str = "unsorted",
        output_dir: str = "sorted",
        width: int = 800,
        height: int = 600,
        window_title: str = "Gems in the Rough",
    ):
        super().__init__()

        self.manager = manager
        self.num_buttons = num_buttons
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.resize(width, height)
        self.setWindowTitle(window_title)

        # I'd like to refactor this to just pop the first element
        self.index = 0
        self.current_image_path: str | None = None
        self._pixmap: QPixmap | None = None

        # region layout
        layout = QVBoxLayout()

        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_display.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        layout.addWidget(self.image_display)

        self.caption = QLabel("Loading...")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setMaximumHeight(40)
        self.caption.setStyleSheet("font-size: 16px;")
        self.caption.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed
        )
        layout.addWidget(self.caption)

        # TODO: can we access the buttons through the QHBoxLayout instead?
        buttons = QHBoxLayout()
        self.move_buttons = [
            Button(
                key=str(i),
                on_click=self.move_image,
                qhbox=buttons,
            )
            for i in range(1, num_buttons + 1)
        ]

        # this is basically a constant button that never changes
        skip_key = "Space"
        self.skip_button = QPushButton(f"Skip ({skip_key})")
        self.skip_button.clicked.connect(self.next)
        self.skip_button.setShortcut(skip_key)
        buttons.addWidget(self.skip_button)

        layout.addLayout(buttons)
        # endregion

        self.setLayout(layout)
        self.manager.result_ready.connect(self.on_result)
        self.manager.error_ready.connect(self.on_error)

        self.load()

    def load(self):
        """
        Load the current image and display it, along with any cached classification results.
        """
        # If we've gone through all the images, show a completion message
        if self.index >= len(self.manager.image_paths):
            self.caption.setText("All done :)")
            self.image_display.clear()
            self.current_image_path = None
            self._pixmap = None
            self.set_thinking()
            return

        self.current_image_path = self.manager.image_paths[self.index]
        self.caption.setText(
            f"{self.current_image_path} ({self.index + 1} / {len(self.manager.image_paths)})"
        )

        # actually display the image here
        self._pixmap = QPixmap(self.current_image_path)
        self._rescale()

        if self.current_image_path in self.manager.cache:
            # if we already have a result for this image, show it immediately
            self.show_result(self.manager.cache[self.current_image_path])
        elif self.current_image_path in self.manager.errors:
            self.show_error(self.manager.errors[self.current_image_path])
        else:
            # otherwise, show a loading message and submit the image for processing
            self.set_thinking()
            self.manager.submit(self.current_image_path)

        self.manager.update_window(self.index)

    def _rescale(self):
        if self._pixmap is None or self._pixmap.isNull():
            self.image_display.clear()
            return
        self.image_display.setPixmap(
            self._pixmap.scaled(
                self.image_display.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def on_result(self, path, labels):
        # do nothing if a result for a previous item or future item comes in
        if path == self.current_image_path:
            self.show_result(labels)

    def on_error(self, path, message):
        if path == self.current_image_path:
            self.show_error(message)

    def set_thinking(self):
        for button in self.move_buttons:
            button.set_thinking()

    def show_result(self, labels):
        for button in self.move_buttons:
            button.set_unavailable()
        for label, button in zip(labels, self.move_buttons, strict=False):
            button.show_result(label)

    def show_error(self, message: str):
        """Leave the image skippable when the classifier could not label it."""
        logger.warning("Could not classify %s: %s", self.current_image_path, message)
        for button in self.move_buttons:
            button.set_unavailable("Failed")

    def move_image(self, destination_dir: str):
        if self.current_image_path is None:
            return

        destination_dir = os.path.join(self.output_dir, destination_dir)
        os.makedirs(destination_dir, exist_ok=True)

        file_destination = os.path.join(
            destination_dir, os.path.basename(self.current_image_path)
        )
        logger.info("Moving %s to %s", self.current_image_path, destination_dir)
        shutil.move(self.current_image_path, file_destination)

        self.next()

    def next(self):
        # might want to refactor this to just pop the previous image
        self.index += 1
        self.load()

    def resizeEvent(self, event):
        self._rescale()
        super().resizeEvent(event)

    def closeEvent(self, event):
        self.manager.cancel_all()
        super().closeEvent(event)
