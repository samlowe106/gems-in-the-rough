from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

Labels = list[str]


class WorkerSignals(QObject):
    result = Signal(str, Labels)  # path, labels


class InferenceTask(QRunnable):
    def __init__(self, path: str, classifier: Callable, folder_names: list[str]):
        super().__init__()
        self.path = path
        self.classifier = classifier
        self.folder_names = folder_names
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        if self._cancelled:
            return

        labels = self.classifier(self.path, categories=self.folder_names)

        if self._cancelled:
            return

        self.signals.result.emit(str(self.path), labels)


class InferenceManager(QObject):
    result_ready = Signal(str, Labels)  # path, labels

    def __init__(
        self,
        classifier: Callable,
        image_paths: list[str],
        max_workers=2,
        lookahead=8,
    ):
        super().__init__()
        self.classifier = classifier
        self.image_paths = image_paths
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(max_workers)

        self.lookahead = lookahead
        self.cache: dict[str, Labels] = {}
        self.in_flight: dict[str, InferenceTask] = {}
        self.queue: list[str] = []

        self.current_index = 0

    def submit(self, path):
        if path in self.cache:
            print("Already have result for:", path)
            return
        if path in self.in_flight:
            print("Already in-flight for:", path)
            return

        task = InferenceTask(path, self.classifier)
        self.in_flight[path] = task

        task.signals.result.connect(self._on_result)
        self.pool.start(task)

    def _on_result(self, path, labels: Labels):
        self.cache[path] = labels
        self.in_flight.pop(path, None)
        self.result_ready.emit(path, labels)

    def update_window(self, current_index):
        self.current_index = current_index

        start = current_index
        end = min(len(self.image_paths), current_index + self.lookahead)

        for i in range(start + 1, end):
            self.submit(self.image_paths[i])

    def cancel_all(self):
        for task in self.in_flight.values():
            task.cancel()
