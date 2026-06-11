# 💎 Gems in the Rough

Gems in the Rough uses a local AI assistant, Gemma, to help you sort images on your computer with human oversight. You specify an input directory and folder options, and a UI will open showing each image in the directory, Gemma's suggested folder, and Gemma's confidence. You can accept Gemma's suggestion and move the file by hitting Y, or you can reject the suggestion by hitting N.

Gems in the Rough features dynamic queueing and loads images as you move through them; you can tweak the number of active threads and how far ahead the program will look before it pauses for you to catch up. You can also choose which version of Gemma 4 you want; this uses the smallest Gemma 4 model, [E2B-it](https://huggingface.co/google/gemma-4-E2B-it), by default.

Made with huggingface and PySide6.

## Installation

Make sure you have a compatible version of Python as specified in the [pyproject.toml file](pyproject.toml) (I used 3.13.9) and that you've installed [uv](https://docs.astral.sh/uv/) to manage the dependencies. From there, running `uv sync` should be enough.
