import os
import shutil
from typing import Callable

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


class Button(QPushButton):
    def __init__(self, shortcut: str, on_click: Callable, qhbox: QHBoxLayout):
        super().__init__()

        self.shortcut = shortcut

        self._on_click_template = on_click
        self._on_click: Callable = None

        super().setText("Thinking...")

        qhbox.addWidget(self)

    def _disconnect_button(self):
        if self._on_click is not None:
            super().clicked.disconnect(self._on_click)

    def set_thinking(self):
        super().setText("Thinking...")
        self._disconnect_button()
        self._on_click = None
        self.setShortcut("")

    def show_result(self, result_label: str, destination_dir=None):
        if destination_dir is None:
            destination_dir = result_label
        super().setText(f"{result_label} ({self.shortcut})")
        self._disconnect_button()
        self.setShortcut(self.shortcut)
        self._on_click = lambda: self._on_click_template(
            destination_dir=destination_dir
        )
        self.clicked.connect(self._on_click)


class ImageSorterUI(QWidget):
    def __init__(
        self,
        manager,
        num_buttons: int,
        input_dir="unsorted",
        output_dir="sorted",
        width=800,
        height=600,
        window_title="Gems in the Rough",
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

        # region layout
        layout = QVBoxLayout()

        self.image_display = QLabel()
        self.image_display.setAlignment(Qt.AlignCenter)
        self.image_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        layout.addWidget(self.image_display)

        self.caption = QLabel("Loading...")
        self.caption.setAlignment(Qt.AlignCenter)
        self.caption.setMaximumHeight(40)
        self.caption.setStyleSheet("font-size: 16px;")
        self.current_label = None
        self.caption.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        layout.addWidget(self.caption)

        # TODO: can we access the buttons through the QHBoxLayout instead?
        buttons = QHBoxLayout()
        self.move_buttons = [
            Button(
                shortcut=str(i),
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
            self.set_thinking()
            return

        self.current_image_path = self.manager.image_paths[self.index]
        self.caption.setText(
            f"{self.current_image_path} ({self.index + 1} / {len(self.manager.image_paths)})"
        )

        # actually display the image here
        pixmap = QPixmap(self.current_image_path)
        self.image_display.setPixmap(
            pixmap.scaled(
                self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
        )

        if self.current_image_path in self.manager.cache:
            # if we already have a result for this image, show it immediately
            labels = self.manager.cache[self.current_image_path]
            self.show_result(labels)
        else:
            # otherwise, show a loading message and submit the image for processing
            self.set_thinking()
            self.manager.submit(self.current_image_path)

        self.manager.update_window(self.index)

    def on_result(self, path, labels):
        # do nothing if a result for a previous item or future item comes in
        if path == self.manager.image_paths[self.index]:
            self.show_result(labels)

    def set_thinking(self):
        for button in self.move_buttons:
            button.set_thinking()

    def show_result(self, labels):
        for label, button in zip(labels, self.move_buttons):
            button.show_result(label)

    def move_image(self, destination_dir: str):
        destination_dir = os.path.join(self.output_dir, destination_dir)
        os.makedirs(destination_dir, exist_ok=True)

        file_destination = os.path.join(
            destination_dir, os.path.basename(self.current_image_path)
        )
        print(f"Moving {self.current_image_path} to {destination_dir}")
        shutil.move(self.current_image_path, file_destination)

        self.next()

    def next(self):
        # might want to refactor this to just pop the previous image
        self.index += 1
        self.load()

    def resizeEvent(self, event):
        if self.current_image_path:
            pixmap = QPixmap(self.current_image_path)
            scaled = pixmap.scaled(
                self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.image_display.setPixmap(scaled)

        super().resizeEvent(event)
