import logging
from collections.abc import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal, Slot

logger = logging.getLogger(__name__)

# Signal doesn't accept this as a type because it's
# a GenericAlias instead of a type
# Labels = list[str]


class WorkerSignals(QObject):
    result = Signal(str, list)  # path, labels
    error = Signal(str, str)  # path, message


class InferenceTask(QRunnable):
    def __init__(self, path: str, classifier: Callable):
        super().__init__()
        self.path = path
        self.classifier = classifier
        self.signals = WorkerSignals()
        self._cancelled = False

    def cancel(self):
        self._cancelled = True

    @Slot()
    def run(self):
        if self._cancelled:
            return

        # An exception escaping QRunnable.run crosses back into C++ and takes the
        # process with it, so every failure has to be caught and reported here.
        try:
            labels = self.classifier(self.path)
        except Exception as exc:
            logger.exception("Classification failed for %s", self.path)
            if not self._cancelled:
                self.signals.error.emit(str(self.path), str(exc))
            return

        if self._cancelled:
            return

        self.signals.result.emit(str(self.path), labels)


class InferenceManager(QObject):
    result_ready = Signal(str, list)  # path, labels
    error_ready = Signal(str, str)  # path, message

    def __init__(
        self,
        classifier: Callable,
        image_paths: list[str],
        max_workers: int = 2,
        lookahead: int = 8,
    ):
        super().__init__()
        self.classifier = classifier
        self.image_paths = image_paths
        self.pool = QThreadPool()
        self.pool.setMaxThreadCount(max_workers)

        self.lookahead = lookahead
        self.cache: dict[str, list[str]] = {}
        self.errors: dict[str, str] = {}
        self.in_flight: dict[str, InferenceTask] = {}

        self.current_index = 0

    def submit(self, path: str) -> None:
        if path in self.cache:
            logger.debug("Already have a result for %s", path)
            return
        if path in self.in_flight:
            logger.debug("Already in flight: %s", path)
            return

        task = InferenceTask(path, self.classifier)
        self.in_flight[path] = task

        task.signals.result.connect(self._on_result)
        task.signals.error.connect(self._on_error)
        self.pool.start(task)

    def _on_result(self, path: str, labels: list[str]) -> None:
        self.cache[path] = labels
        self.in_flight.pop(path, None)
        self.result_ready.emit(path, labels)

    def _on_error(self, path: str, message: str) -> None:
        self.errors[path] = message
        self.in_flight.pop(path, None)
        self.error_ready.emit(path, message)

    def update_window(self, current_index: int) -> None:
        """Prefetch classifications for the images just past `current_index`."""
        self.current_index = current_index

        end = min(len(self.image_paths), current_index + self.lookahead)

        for i in range(current_index + 1, end):
            self.submit(self.image_paths[i])

    def cancel_all(self) -> None:
        for task in self.in_flight.values():
            task.cancel()
        self.pool.clear()
        self.pool.waitForDone()
