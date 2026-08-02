import os

import pytest

from src.main import (
    DEFAULT_FOLDERS,
    build_classifier,
    find_images,
    manual_classifier,
    parse_args,
    validate,
)


def args_for(image_dir: str, **overrides):
    args = parse_args(["--input-dir", image_dir])
    for key, value in overrides.items():
        setattr(args, key, value)
    if args.folders is None:
        args.folders = list(DEFAULT_FOLDERS)
    return args


class TestFindImages:
    def test_returns_only_supported_image_files(self, image_dir):
        found = find_images(image_dir)
        assert [os.path.basename(path) for path in found] == [
            "a.png",
            "b.jpg",
            "c.JPEG",
        ]

    def test_extension_match_is_case_insensitive(self, image_dir):
        assert any(path.endswith("c.JPEG") for path in find_images(image_dir))

    def test_skips_directories_that_look_like_images(self, image_dir):
        assert all("nested.png" not in path for path in find_images(image_dir))

    def test_empty_directory_yields_no_paths(self, tmp_path):
        assert find_images(str(tmp_path)) == []


class TestManualClassifier:
    def test_returns_the_first_k_categories_in_order(self, folders):
        assert manual_classifier(folders, 3) == folders[:3]

    def test_does_not_alias_the_caller_list(self, folders):
        result = manual_classifier(folders, 5)
        result.append("Other")
        assert len(folders) == 5


class TestBuildClassifier:
    def test_manual_mode_needs_no_model(self, folders):
        classifier = build_classifier("manual", folders, 2)
        assert classifier("/does/not/exist.png") == folders[:2]


class TestValidate:
    def test_accepts_the_defaults(self, image_dir):
        validate(args_for(image_dir))

    def test_rejects_unknown_mode(self, image_dir):
        with pytest.raises(ValueError, match="Mode must be one of"):
            validate(args_for(image_dir, mode="E9B-it"))

    @pytest.mark.parametrize("top", [0, -1, 6])
    def test_rejects_top_outside_the_folder_count(self, image_dir, top):
        with pytest.raises(ValueError, match="Top K"):
            validate(args_for(image_dir, top=top))

    def test_rejects_top_above_the_single_digit_shortcut_limit(self, image_dir):
        folders = [f"Folder{i}" for i in range(12)]
        with pytest.raises(ValueError, match="Top K"):
            validate(args_for(image_dir, folders=folders, top=10))

    def test_allows_top_equal_to_the_shortcut_limit(self, image_dir):
        folders = [f"Folder{i}" for i in range(12)]
        validate(args_for(image_dir, folders=folders, top=9))

    @pytest.mark.parametrize("field", ["max_workers", "lookahead"])
    def test_rejects_non_positive_concurrency_settings(self, image_dir, field):
        with pytest.raises(ValueError, match="at least 1"):
            validate(args_for(image_dir, **{field: 0}))

    def test_rejects_missing_input_directory(self, tmp_path):
        missing = str(tmp_path / "nope")
        with pytest.raises(ValueError, match="does not exist"):
            validate(args_for(missing))


class TestParseArgs:
    def test_defaults(self):
        args = parse_args([])
        assert args.mode == "E2B-it"
        assert args.top == 4
        assert args.max_workers == 2
        assert args.lookahead == 8
        assert args.folders is None
        assert args.verbose is False

    def test_folders_take_multiple_values(self):
        assert parse_args(["-f", "Cats", "Dogs"]).folders == ["Cats", "Dogs"]

    def test_unknown_mode_is_rejected_at_the_parser(self):
        with pytest.raises(SystemExit):
            parse_args(["--mode", "gpt"])
