import os

import pytest
from PIL import Image

FOLDERS = ["Art", "Memes", "Nature", "People", "Screenshots"]


def write_image(path: str, color: tuple[int, int, int] = (120, 90, 200)) -> str:
    """Write a tiny real image so PIL and QPixmap both accept it."""
    Image.new("RGB", (16, 16), color).save(path)
    return path


@pytest.fixture
def folders() -> list[str]:
    return list(FOLDERS)


@pytest.fixture
def image_dir(tmp_path) -> str:
    """A directory holding three images plus decoys the scanner must ignore."""
    directory = tmp_path / "unsorted"
    directory.mkdir()

    for i, name in enumerate(["a.png", "b.jpg", "c.JPEG"]):
        write_image(str(directory / name), (i * 40, 90, 200))

    # Not images, and a subdirectory: none of these should be picked up.
    (directory / "notes.txt").write_text("not an image")
    (directory / "clip.mp4").write_bytes(b"\x00\x00\x00\x18ftyp")
    (directory / "nested.png").mkdir()

    return str(directory)


@pytest.fixture
def output_dir(tmp_path) -> str:
    directory = tmp_path / "sorted"
    directory.mkdir()
    return str(directory)


@pytest.fixture
def image_paths(image_dir) -> list[str]:
    return [
        os.path.join(image_dir, name)
        for name in sorted(os.listdir(image_dir))
        if name.lower().endswith((".png", ".jpg", ".jpeg"))
        and os.path.isfile(os.path.join(image_dir, name))
    ]
