import logging
import os
import sys
from argparse import ArgumentParser, Namespace

from PySide6.QtWidgets import QApplication

from .inference_manager import InferenceManager
from .models import Gemma
from .ui import ImageSorterUI

logger = logging.getLogger(__name__)

IMAGE_FILE_EXTENSIONS = (".png", ".jpg", ".jpeg")
ACCEPTED_MODES = ("manual", "E2B-it", "E4B-it", "4-31B-it")
DEFAULT_FOLDERS = ("Art", "Memes", "Nature", "People", "Screenshots")

# At most 9 so every label maps to a single-digit keyboard shortcut.
MAX_TOP_K = 9


def manual_classifier(categories: list[str], top_k: int) -> list[str]:
    """Offer the first `top_k` categories unchanged, with no model in the loop."""
    return list(categories[:top_k])


def find_images(input_dir: str) -> list[str]:
    """Return the paths of every supported image file directly inside `input_dir`."""
    return [
        os.path.join(input_dir, filename)
        for filename in sorted(os.listdir(input_dir))
        if os.path.isfile(os.path.join(input_dir, filename))
        and filename.lower().endswith(IMAGE_FILE_EXTENSIONS)
    ]


def build_classifier(mode: str, folders: list[str], top_k: int):
    """Build the callable that maps an image path to a ranked list of categories."""
    if mode == "manual":

        def classify(_image_path: str) -> list[str]:
            return manual_classifier(folders, top_k)

        return classify

    return Gemma(model_name=mode, categories=folders, top_k=top_k).classify


def validate(args: Namespace) -> None:
    if args.mode not in ACCEPTED_MODES:
        raise ValueError(f"Mode must be one of {list(ACCEPTED_MODES)}")

    if not 0 < args.top <= min(len(args.folders), MAX_TOP_K):
        raise ValueError(
            "Top K must be between 1 and the number of folders (inclusive), "
            f"and at most {MAX_TOP_K}."
        )

    if args.max_workers < 1:
        raise ValueError("--max-workers must be at least 1")

    if args.lookahead < 1:
        raise ValueError("--lookahead must be at least 1")

    if not os.path.isdir(args.input_dir):
        raise ValueError(f"Input directory does not exist: {args.input_dir}")


def main(args: Namespace) -> int:
    if not args.folders:
        args.folders = list(DEFAULT_FOLDERS)

    validate(args)

    logger.info("Folder options: %s", args.folders)
    logger.info("Reading from %s, writing to %s", args.input_dir, args.output_dir)

    classifier = build_classifier(args.mode, args.folders, args.top)

    image_paths = find_images(args.input_dir)
    logger.info("Found %d images.", len(image_paths))

    app = QApplication(sys.argv)

    manager = InferenceManager(
        classifier=classifier,
        image_paths=image_paths,
        max_workers=args.max_workers,
        lookahead=args.lookahead,
    )

    ui = ImageSorterUI(
        manager,
        num_buttons=args.top,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    ui.show()

    # aboutToQuit fires for a closed window, a KeyboardInterrupt, and a session
    # logout alike, so every exit path drains the thread pool exactly once.
    app.aboutToQuit.connect(manager.cancel_all)

    return app.exec()


def parse_args(argv: list[str] | None = None) -> Namespace:
    parser = ArgumentParser(
        prog="gems-in-the-rough",
        description="Sort images into folders with a local Gemma model and human oversight.",
    )
    parser.add_argument(
        "--input-dir", help="Path to the images to be sorted", default="tests/unsorted"
    )
    parser.add_argument(
        "--output-dir",
        help="Path where the sorted images will be moved to",
        default="tests/sorted",
    )
    parser.add_argument(
        "-m",
        "--mode",
        help="Which classifier to use",
        default="E2B-it",
        choices=ACCEPTED_MODES,
    )
    parser.add_argument(
        "--top", help="How many top categories to return", type=int, default=4
    )
    parser.add_argument(
        "--max-workers",
        help="Number of concurrent inference threads",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--lookahead", help="Number of images to pre-classify", type=int, default=8
    )
    parser.add_argument(
        "-f", "--folders", help="The folder names to choose from", nargs="+"
    )
    parser.add_argument(
        "-v", "--verbose", help="Log every classification", action="store_true"
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    parsed = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if parsed.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    sys.exit(main(parsed))
