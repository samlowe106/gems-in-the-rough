import re

from PIL import Image

# transformers, torchvision, accelerate
from transformers import AutoModelForMultimodalLM, AutoProcessor


class Gemma:

    def __init__(self, model_name="E2B-it"):
        # https://huggingface.co/google/gemma-4-E2B-it

        self.name = f"google/gemma-4-{model_name}"

        self._processor = AutoProcessor.from_pretrained(self.name)
        self._model = AutoModelForMultimodalLM.from_pretrained(
            self.name, dtype="auto", device_map="auto", local_files_only=True
        )

    def create_prompt(self, image_filepath: str, folder_names: list[str]) -> list[dict]:
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
                        .resize((384, 384)),
                    },
                    {
                        "type": "text",
                        "text": (
                            "Which folder would best suit this image?"
                            "Your response should be exactly of the form <FOLDER_NAME> <CONFIDENCE>"
                            f"where <FOLDER_NAME> is one of {folder_names}"
                            "and <CONFIDENCE> is a float between 0 and 1"
                            "representing how confident you are in your classification."
                        ),
                    },
                ],
            },
        ]

    # TODO: this method can have a lot more optional parameters for controlling the generation,
    # e.g. temperature, max_new_tokens, etc.
    def _generate_response(self, image_filepath: str, folder_names: list[str]) -> str:
        # https://huggingface.co/google/gemma-4-31B-it#5-variable-image-resolution
        # The supported token budgets are: 70, 140, 280, 560, and 1120.
        # Use lower budgets for classification, captioning, or video understanding,
        # where faster inference and processing many frames outweigh fine-grained detail
        # Use higher budgets for tasks like OCR, document parsing, or reading small text.
        inputs = self._processor.apply_chat_template(
            self.create_prompt(image_filepath, folder_names),
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
        ).to(self._model.device)
        input_len = inputs["input_ids"].shape[-1]

        # https://huggingface.co/docs/transformers/llm_tutorial#common-options
        # Temperature: How unpredictable the next selected token will be.
        # High values (>0.8) are good for creative tasks,
        # low values (e.g. <0.4) for tasks that require “thinking”. Requires do_sample=True.
        outputs = self._model.generate(
            **inputs, temperature=0.2, do_sample=True, max_new_tokens=100
        )
        response = self._processor.decode(
            outputs[0][input_len:], skip_special_tokens=False
        )

        # Parse output
        self._processor.parse_response(response)
        return response

    def _validate_response(self, response: str, folder_names: list[str]) -> bool:
        pattern = re.compile(rf"^({'|'.join(folder_names)})\s(0(\.\d+)?|1(\.0+)?)$")
        return bool(pattern.match(response))

    def classify(
        self, image_filepath: str, folder_names: list[str]
    ) -> tuple[str, float]:

        response = self._generate_response(image_filepath, folder_names)

        if not self._validate_response(response, folder_names):
            raise ValueError("Invalid response format")

        folder_name, confidence = response.split(" ")
        return folder_name, float(confidence)
