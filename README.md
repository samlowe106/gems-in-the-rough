# 💎 Gems in the Rough

[![Run Tests](https://github.com/samlowe106/gems-in-the-rough/actions/workflows/tests.yml/badge.svg)](https://github.com/samlowe106/gems-in-the-rough/actions/workflows/tests.yml)

Gems in the Rough uses a local AI assistant, Gemma, to help you sort images on your computer with human oversight. You specify an input directory and a set of folder names, and a UI opens showing each image alongside Gemma's top few suggested folders. Press the number key next to a suggestion to move the file into that folder, or press Space to skip the image and leave it where it is.

Gems in the Rough features dynamic queueing and classifies images ahead of you as you work; you can tweak the number of active threads and how far ahead the program looks. You can also choose which version of Gemma 4 you want; this uses the smallest Gemma 4 model, [E2B-it](https://huggingface.co/google/gemma-4-E2B-it), by default. Nothing leaves your machine: the Hugging Face libraries are pinned to local files only.

Made with huggingface and PySide6.

## Installation

Install [uv](https://docs.astral.sh/uv/), then:

```bash
uv sync             # create .venv and install everything from uv.lock
pre-commit install  # run the hooks automatically on every commit
```

uv provisions the Python version pinned in [.python-version](.python-version), so you do not need a matching system interpreter.

The Gemma weights are not downloaded for you. Fetch the model you plan to use once (for example `hf download google/gemma-4-E2B-it`) before running with a non-manual mode.

## Usage

```bash
uv run python -m src.main --input-dir ~/Pictures/unsorted --output-dir ~/Pictures/sorted
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--input-dir` | `tests/unsorted` | Directory of images to sort. Not searched recursively. |
| `--output-dir` | `tests/sorted` | Where the chosen folders are created. |
| `-f`, `--folders` | `Art Memes Nature People Screenshots` | Candidate folder names. |
| `--top` | `4` | How many suggestions to offer, at most 9 so each maps to a single-digit key. |
| `-m`, `--mode` | `E2B-it` | `manual`, `E2B-it`, `E4B-it`, or `4-31B-it`. |
| `--max-workers` | `2` | Concurrent inference threads. |
| `--lookahead` | `8` | How many images ahead to pre-classify. |
| `-v`, `--verbose` | off | Log each classification and each raw model response. |

`--mode manual` skips the model entirely and just offers the first `--top` folders, which is the fastest way to try the UI or to sort by hand.

## Development

```bash
uv run pytest                      # tests with coverage and per-test timings
uv run pytest -m "not model"       # skip anything that loads real weights
uv run pre-commit run --all-files  # every hook manually
```

Qt needs a display. On a headless machine, set `QT_QPA_PLATFORM=offscreen`, which is what CI does.

### Layout

```
.
├── .github/workflows/    # CI: pre-commit, pytest, pip-audit, CodeQL
├── src/
│   ├── main.py           # CLI entry point and argument validation
│   ├── ui.py             # PySide6 widgets and the file-moving logic
│   ├── inference_manager.py  # QThreadPool queueing, caching, and lookahead
│   └── models/gemma.py   # prompt construction, generation, response validation
└── tests/                # pytest suite (no model weights required)
```

### Tooling

Linting, formatting, and type-checking run on every commit via [pre-commit](https://pre-commit.com/): [Ruff](https://docs.astral.sh/ruff/) for lint and import sorting, [Black](https://black.readthedocs.io/) for formatting, and [ty](https://github.com/astral-sh/ty) for type-checking. All three read their configuration from [pyproject.toml](pyproject.toml), so your editor, the CLI, and CI agree. Hook versions are refreshed monthly by the [pre-commit-autoupdate](.github/workflows/pre-commit-autoupdate.yml) workflow.

Ruff's `T20` rule flags stray `print()` calls, so user-facing output goes through `logging` instead.
