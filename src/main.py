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


def main(args):
    accepted_models = ["E2B-it", "E4B-it", "4-31B-it"]

    if args.model not in accepted_models:
        raise ValueError(f"Model must be one of {accepted_models}")
    model = Gemma(model_name=args.model)

    app = QApplication(sys.argv)

    print("Selected directory:", args.input_path)

    image_paths = [
        os.path.join(args.input_path, filename)
        for filename in os.listdir(args.input_path)
        if os.path.isfile(os.path.join(args.input_path, filename))
        and filename.lower().endswith(image_file_extensions)
    ]

    print(f"Found {len(image_paths)} images.")

    if not args.folders:
        args.folders = ["Art", "Memes", "Nature", "People", "Screenshots"]

    print("Folder options:", args.folders)

    manager = InferenceManager(
        model_fn=model.classify,
        image_paths=image_paths,
        folder_names=args.folders,
        max_workers=1,
        lookahead=8,
    )

    ui = ImageSorterUI(manager)
    ui.show()
    ui.load()
    app.exec()


if __name__ == "__main__":

    print(os.getcwd())

    # region Argument parsing
    parser = ArgumentParser()
    parser.add_argument(
        "--input-path", help="Path to the images to be sorted", default="tests/unsorted"
    )
    parser.add_argument(
        "--output-path", help="Path where the sorted images will be moved to"
    )
    parser.add_argument(
        "-m", "--model", help="Which Gemma model to use", default="E2B-it"
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
