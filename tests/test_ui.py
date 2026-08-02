import os

import pytest

from src.inference_manager import InferenceManager
from src.ui import ImageSorterUI


@pytest.fixture
def ui(qtbot, qapp, image_paths, image_dir, output_dir, folders):
    manager = InferenceManager(
        classifier=lambda path: folders[:2], image_paths=image_paths, max_workers=1
    )
    widget = ImageSorterUI(
        manager,
        num_buttons=2,
        input_dir=image_dir,
        output_dir=output_dir,
    )
    qtbot.addWidget(widget)
    yield widget
    manager.cancel_all()


def test_starts_on_the_first_image(ui, image_paths):
    assert ui.index == 0
    assert ui.current_image_path == image_paths[0]
    assert "1 / 3" in ui.caption.text()


def test_thinking_state_offers_nothing_to_click(ui):
    # The stub classifier returns instantly, so drive the state directly rather
    # than racing the worker thread.
    ui.set_thinking()

    assert all(button.text() == "Thinking..." for button in ui.move_buttons)
    assert all(not button.isEnabled() for button in ui.move_buttons)


def test_labels_become_buttons_once_a_result_arrives(qtbot, ui, folders):
    qtbot.waitUntil(lambda: ui.move_buttons[0].isEnabled(), timeout=5000)

    assert ui.move_buttons[0].text() == f"{folders[0]} (1)"
    assert ui.move_buttons[1].text() == f"{folders[1]} (2)"


def test_a_result_for_another_image_is_ignored(ui, image_paths):
    ui.on_result(image_paths[2], ["Nature", "People"])
    assert all("Nature" not in button.text() for button in ui.move_buttons)


def test_clicking_a_label_moves_the_file_and_advances(qtbot, ui, output_dir, folders):
    qtbot.waitUntil(lambda: ui.move_buttons[0].isEnabled(), timeout=5000)
    source = ui.current_image_path

    ui.move_buttons[0].click()

    moved_to = os.path.join(output_dir, folders[0], os.path.basename(source))
    assert os.path.isfile(moved_to)
    assert not os.path.exists(source)
    assert ui.index == 1


def test_skip_advances_without_touching_the_file(qtbot, ui):
    source = ui.current_image_path

    ui.skip_button.click()

    assert os.path.isfile(source)
    assert ui.index == 1


def test_running_off_the_end_shows_the_done_state(ui, image_paths):
    for _ in image_paths:
        ui.next()

    assert ui.current_image_path is None
    assert ui.caption.text() == "All done :)"
    assert all(not button.isEnabled() for button in ui.move_buttons)


def test_a_failed_classification_leaves_the_image_skippable(qtbot, ui):
    ui.on_error(ui.current_image_path, "Invalid response")

    assert all(button.text() == "Failed" for button in ui.move_buttons)
    assert all(not button.isEnabled() for button in ui.move_buttons)
    assert ui.skip_button.isEnabled()


def test_moving_with_no_current_image_is_a_no_op(ui, output_dir, folders):
    ui.current_image_path = None
    ui.move_image(folders[0])

    assert not os.path.exists(os.path.join(output_dir, folders[0]))
