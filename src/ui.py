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
    def __init__(self, manager, width=800, height=600):
        super().__init__()

        self.resize(width, height)

        self.manager = manager
        self.index = 0

        self.label = QLabel()
        self.label.setAlignment(Qt.AlignCenter)

        self.input_dir = "unsorted"
        self.output_dir = "sorted"

        self.suggestion = QLabel("Loading...")
        self.suggestion.setAlignment(Qt.AlignCenter)
        self.suggestion.setMaximumHeight(40)
        self.suggestion.setStyleSheet("font-size: 16px;")
        self.current_label = None

        self.label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.suggestion.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)

        accept_key = "Y"
        skip_key = "N"

        self.accept_button = QPushButton(f"Accept ({accept_key})")
        self.skip_button = QPushButton(f"Skip ({skip_key})")

        self.accept_button.clicked.connect(self.accept_image)
        self.skip_button.clicked.connect(self.next)

        self.accept_button.setShortcut(accept_key)
        self.skip_button.setShortcut(skip_key)

        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.suggestion)

        buttons = QHBoxLayout()
        buttons.addWidget(self.accept_button)
        buttons.addWidget(self.skip_button)

        layout.addLayout(buttons)
        self.setLayout(layout)

        self.manager.result_ready.connect(self.on_result)

    def load(self):
        if self.index >= len(self.manager.image_paths):
            self.suggestion.setText("All done!")
            self.label.clear()
            return

        path = self.manager.image_paths[self.index]
        self.current_image_path = path

        pixmap = QPixmap(path)

        scaled = pixmap.scaled(
            self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

        self.label.setPixmap(scaled)

        if path in self.manager.cache:
            label, conf = self.manager.cache[path]
            self.show_result(label, conf)
        else:
            self.suggestion.setText("Thinking...")
            self.manager.submit(path)

        self.manager.update_window(self.index)

    def on_result(self, path, label, conf):
        if path == self.manager.image_paths[self.index]:
            self.show_result(label, conf)

    def show_result(self, label, conf):
        self.current_label = label
        self.suggestion.setText(f"{label} ({conf:.0%})")

    def accept_image(self):
        if not self.current_label:
            return

        destination_dir = os.path.join(self.output_dir, self.current_label)
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
        if hasattr(self, "current_image_path"):
            pixmap = QPixmap(self.current_image_path)

            scaled = pixmap.scaled(
                self.label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )

            self.label.setPixmap(scaled)

        super().resizeEvent(event)
