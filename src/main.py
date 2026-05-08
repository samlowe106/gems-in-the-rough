import os
import sys
from argparse import ArgumentParser

from PySide6.QtWidgets import (
    QApplication,
)

from inference_manager import InferenceManager
from models import Gemma
from ui import ImageSorterUI

image_file_extensions = (".png", ".jpg", ".jpeg")
accepted_modes = ["manual", "E2B-it", "E4B-it", "4-31B-it"]


def manual_classifier(categories: list[str], top_k: int, *args, **kwargs) -> list[str]:
    return categories[:top_k]


def main(args):

    if not args.folders:
        args.folders = ["Art", "Memes", "Nature", "People", "Screenshots"]

    print("Folder options:", args.folders)

    if not 0 < args.top <= min(len(args.folders), 9):
        # we require at most 9 so the labels are single digit numbers for easier keyboard input
        raise ValueError(
            "Top K must be between 1 and the number of folders (inclusive), and at most 9."
        )

    if args.mode not in accepted_modes:
        raise ValueError(f"Mode must be one of {accepted_modes}")

    classifier = (
        (lambda x: manual_classifier(args.folders, args.top))
        if args.mode == "manual"
        else Gemma(
            model_name=args.mode,
            categories=args.folders,
            top_k=args.top,
        ).classify
    )

    app = QApplication(sys.argv)

    print("Selected directory:", args.input_dir)
    print("Will output to:", args.output_dir)

    image_paths = [
        os.path.join(args.input_dir, filename)
        for filename in os.listdir(args.input_dir)
        if os.path.isfile(os.path.join(args.input_dir, filename))
        and filename.lower().endswith(image_file_extensions)
    ]

    print(f"Found {len(image_paths)} images.")

    manager = InferenceManager(
        classifier=classifier,
        image_paths=image_paths,
        max_workers=1,
        lookahead=8,
    )

    ui = ImageSorterUI(
        manager,
        num_buttons=args.top,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
    )
    ui.show()
    ui.load()

    # TODO: if the window closes or a KeyboardInterrupt is received, we should
    #  kill all the threads and clean up the program
    app.exec()


if __name__ == "__main__":

    # region Argument parsing
    parser = ArgumentParser()
    parser.add_argument(
        "--input-dir", help="Path to the images to be sorted", default="tests/unsorted"
    )
    parser.add_argument(
        "--output-dir",
        help="Path where the sorted images will be moved to",
        default="tests/sorted",
    )
    parser.add_argument(
        "-m", "--mode", help="Which classifier to use", default="E2B-it"
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

    args = parser.parse_args()
    # endregion

    main(args)
