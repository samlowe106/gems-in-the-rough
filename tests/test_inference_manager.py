import threading

import pytest

from src.inference_manager import InferenceManager


class RecordingClassifier:
    """A stand-in classifier that remembers every path it was handed."""

    def __init__(self, labels: list[str]):
        self.labels = labels
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def __call__(self, path: str) -> list[str]:
        with self._lock:
            self.calls.append(path)
        return list(self.labels)


@pytest.fixture
def classifier() -> RecordingClassifier:
    return RecordingClassifier(["Art", "Memes"])


@pytest.fixture
def manager(qapp, image_paths, classifier):
    manager = InferenceManager(
        classifier=classifier, image_paths=image_paths, max_workers=1, lookahead=2
    )
    yield manager
    manager.cancel_all()


def test_submit_caches_and_announces_the_result(qtbot, manager, image_paths):
    with qtbot.waitSignal(manager.result_ready, timeout=5000) as blocker:
        manager.submit(image_paths[0])

    assert blocker.args == [image_paths[0], ["Art", "Memes"]]
    assert manager.cache[image_paths[0]] == ["Art", "Memes"]
    assert image_paths[0] not in manager.in_flight


def test_a_cached_path_is_not_classified_twice(qtbot, manager, classifier, image_paths):
    with qtbot.waitSignal(manager.result_ready, timeout=5000):
        manager.submit(image_paths[0])

    manager.submit(image_paths[0])
    qtbot.wait(50)

    assert classifier.calls.count(image_paths[0]) == 1


def test_a_classifier_failure_is_reported_rather_than_raised(qtbot, qapp, image_paths):
    def classifier(path: str) -> list[str]:
        raise ValueError("Invalid response")

    manager = InferenceManager(classifier=classifier, image_paths=image_paths)

    with qtbot.waitSignal(manager.error_ready, timeout=5000) as blocker:
        manager.submit(image_paths[0])

    assert blocker.args[0] == image_paths[0]
    assert "Invalid response" in blocker.args[1]
    assert manager.errors[image_paths[0]] == blocker.args[1]
    assert image_paths[0] not in manager.cache
    manager.cancel_all()


def test_update_window_prefetches_the_lookahead_but_not_the_current_image(
    qtbot, manager, classifier, image_paths
):
    manager.update_window(0)
    qtbot.wait(200)

    # lookahead=2 means only the single image after the current one.
    assert classifier.calls == [image_paths[1]]


def test_update_window_stops_at_the_end_of_the_list(
    qtbot, manager, classifier, image_paths
):
    manager.update_window(len(image_paths) - 1)
    qtbot.wait(100)

    assert classifier.calls == []


def test_cancel_all_marks_in_flight_tasks_cancelled(qapp, image_paths):
    started = threading.Event()
    release = threading.Event()

    def classifier(path: str) -> list[str]:
        started.set()
        release.wait(timeout=5)
        return ["Art"]

    manager = InferenceManager(
        classifier=classifier, image_paths=image_paths, max_workers=1
    )
    manager.submit(image_paths[0])
    assert started.wait(timeout=5)

    task = manager.in_flight[image_paths[0]]
    release.set()
    manager.cancel_all()

    assert task._cancelled
    assert image_paths[0] not in manager.cache
