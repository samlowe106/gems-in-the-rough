import os
import sys
from argparse import ArgumentParser

from PySide6.QtWidgets import (
    QApplication,
)

from inference_manager import InferenceManager
from models import Gemma
from ui import ImageSorterUI

# forces Hugging Face libraries to use local files only
os.environ["HF_HUB_OFFLINE"] = "1"


image_file_extensions = (".png", ".jpg", ".jpeg")


def manual_classifier(image_path: str, categories: list[str]) -> list[str]:
    return categories


def main(args):

    accepted_modes = ["manual", "E2B-it", "E4B-it", "4-31B-it"]

    if args.mode not in accepted_modes:
        raise ValueError(f"Mode must be one of {accepted_modes}")

    if args.mode == "manual":
        classifier = manual_classifier
    else:
        classifier = Gemma(model_name=args.model).classify

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

    if not args.folders:
        args.folders = ["Art", "Memes", "Nature", "People", "Screenshots"]

    print("Folder options:", args.folders)

    manager = InferenceManager(
        model_fn=classifier,
        image_paths=image_paths,
        folder_names=args.folders,
        max_workers=1,
        lookahead=8,
    )

    ui = ImageSorterUI(manager, input_dir=args.input_dir, output_dir=args.output_dir)
    ui.show()
    ui.load()
    app.exec()


if __name__ == "__main__":

    print(os.getcwd())

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
