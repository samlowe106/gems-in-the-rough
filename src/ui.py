import os
import shutil

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

        # I would like to eventually not need this
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

        # region buttons
        self.image_display_buttons = []  # do we even need this field?
        buttons = QHBoxLayout()
        for i in range(1, num_buttons + 1):
            button = QPushButton(f"Accept ({i})")
            button.clicked.connect(lambda _, idx=i - 1: self.accept_image(idx))
            button.setShortcut(str(i))
            buttons.addWidget(button)
            self.image_display_buttons.append(button)

        skip_key = "Space"
        self.skip_button = QPushButton(f"Skip ({skip_key})")
        self.skip_button.clicked.connect(self.next)
        self.skip_button.setShortcut(skip_key)
        buttons.addWidget(self.skip_button)
        self.image_display_buttons.append(self.skip_button)

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
            self.show_result(["Thinking..."] * self.num_buttons)
            # self.caption.setText("Thinking...")
            self.manager.submit(self.current_image_path)

        self.manager.update_window(self.index)

    def on_result(self, path, labels):
        # do nothing if a result for a previous item or future item comes in
        if path == self.manager.image_paths[self.index]:
            self.show_result(labels)

    def show_result(self, labels):
        for label, button in zip(labels, self.image_display_buttons):
            button.setText(f"{label} ({button.shortcut().toString()})")

    def accept_image(self, key):
        # TODO: self.folders doesn't exist, get it from the labels or something instead
        destination_dir = os.path.join(self.output_dir, self.folders[key - 1])
        os.makedirs(destination_dir, exist_ok=True)

        file_destination = os.path.join(
            destination_dir, os.path.basename(self.current_image_path)
        )
        shutil.move(self.current_image_path, file_destination)

        self.next()

    def next(self):
        self.index += 1
        self.load()

    def resizeEvent(self, event):
        pixmap = QPixmap(self.current_image_path)

        scaled = pixmap.scaled(
            self.image_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.image_display.setPixmap(scaled)

        # TODO problem: we can't shrink the window if it would shrink the image

        super().resizeEvent(event)
