import logging
import os

from PIL import Image

# transformers, torchvision, accelerate
from transformers import AutoModelForMultimodalLM, AutoProcessor

logger = logging.getLogger(__name__)

Labels = list[str]

# forces Hugging Face libraries to use local files only
os.environ["HF_HUB_OFFLINE"] = "1"


class Gemma:
    def __init__(
        self,
        categories: list[str],
        top_k: int,
        model_name: str = "E2B-it",
        image_size: int = 128,
    ):
        # https://huggingface.co/google/gemma-4-E2B-it

        self.name = f"google/gemma-4-{model_name}"

        self._processor = AutoProcessor.from_pretrained(
            self.name, local_files_only=True
        )
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.name, dtype="auto", device_map="auto", local_files_only=True
        )

        self.categories = categories
        self.top_k = top_k
        self.image_size = image_size

    def create_prompt(
        self,
        image_filepath: str,
        categories: list[str] | None = None,
        top_k: int | None = None,
    ) -> list[dict]:
        if categories is None:
            categories = self.categories
        if top_k is None:
            top_k = self.top_k

        required_format = " ".join(["<category>"] * top_k)
        return [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful assistant that classifies images.",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": Image.open(image_filepath)
                        .convert("RGB")
                        .resize((self.image_size, self.image_size)),
                    },
                    {
                        "type": "text",
                        "text": (
                            f"Give the best {top_k} classifications for the above image"
                            " in descending order of confidence from the following categories:"
                            f" {categories}. Your answer should be in exactly the form:\n"
                            f"{required_format}"
                            "\nseparated by spaces with no punctuation or explanation."
                        ),
                    },
                ],
            },
        ]

    @staticmethod
    def _try_sanitize_output(text: str) -> str:
        """
        Gemma sometimes replies with commas, or repeats a category, even though
        we ask it not to. Strip the punctuation and keep the first occurrence of
        each category, preserving the order it gave them in.
        """
        text = text.replace(",", "")
        seen: set[str] = set()
        options = []
        for item in text.split():
            if item not in seen:
                seen.add(item)
                options.append(item)
        return " ".join(options)

    # TODO: this method can have a lot more optional parameters for controlling the generation,
    # e.g. temperature, max_new_tokens, etc.
    def _generate_response(
        self,
        image_filepath: str,
        categories: list[str] | None = None,
        top_k: int | None = None,
    ) -> str:
        if categories is None:
            categories = self.categories
        if top_k is None:
            top_k = self.top_k

        # https://huggingface.co/google/gemma-4-31B-it#5-variable-image-resolution
        # The supported token budgets are: 70, 140, 280, 560, and 1120.
        # Use lower budgets for classification, captioning, or video understanding,
        # where faster inference and processing many frames outweigh fine-grained detail
        # Use higher budgets for tasks like OCR, document parsing, or reading small text.
        inputs = self._processor.apply_chat_template(
            self.create_prompt(image_filepath, categories, top_k),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        # https://huggingface.co/docs/transformers/llm_tutorial#common-options
        # Temperature: How unpredictable the next selected token will be.
        # High values (>0.8) are good for creative tasks,
        # low values (e.g. <0.4) for tasks that require "thinking". Requires do_sample=True.
        #
        # The answer is top_k category names, so budgeting a token per character of
        # the longest possible answer is a safe over-estimate that still stops the
        # model from rambling.
        outputs = self._model.generate(
            **inputs,
            temperature=0.2,
            do_sample=True,
            max_new_tokens=sum(len(category) for category in categories) + top_k,
        )
        return self._processor.decode(outputs[0][input_len:], skip_special_tokens=True)

    def _validate_response(
        self,
        response: str,
        categories: list[str] | None = None,
        top_k: int | None = None,
    ) -> bool:
        if categories is None:
            categories = self.categories
        if top_k is None:
            top_k = self.top_k

        options = response.split()
        return (
            len(options) == top_k
            and len(set(options)) == len(options)
            and set(options).issubset(set(categories))
        )

    def classify(
        self,
        image_filepath: str,
        categories: list[str] | None = None,
        top_k: int | None = None,
    ) -> Labels:
        if categories is None:
            categories = self.categories
        if top_k is None:
            top_k = self.top_k

        logger.debug("Classifying %s with Gemma...", image_filepath)

        response = self._generate_response(image_filepath, categories, top_k)
        logger.debug("Raw response for %s: %s", image_filepath, response)

        response = self._try_sanitize_output(response)

        if not self._validate_response(response, categories, top_k):
            raise ValueError(
                f"Invalid response for {image_filepath}: {response!r} is not "
                f"{top_k} distinct categories from {categories}"
            )

        return response.split()
