"""
Unit tests for Gemma's prompt building and output handling.

These deliberately bypass `Gemma.__init__`, which downloads and loads several
gigabytes of weights. Anything that needs the real model belongs behind the
`model` marker instead.
"""

import pytest

from src.models.gemma import Gemma


@pytest.fixture
def gemma(folders):
    model = Gemma.__new__(Gemma)
    model.categories = folders
    model.top_k = 3
    model.image_size = 128
    return model


class TestSanitizeOutput:
    def test_strips_commas(self):
        assert Gemma._try_sanitize_output("Art, Memes, Nature") == "Art Memes Nature"

    def test_drops_repeats_but_keeps_first_position(self):
        assert Gemma._try_sanitize_output("Art Memes Art Nature") == "Art Memes Nature"

    def test_collapses_runs_of_whitespace(self):
        assert Gemma._try_sanitize_output("Art   Memes\nNature") == "Art Memes Nature"

    def test_leaves_a_clean_response_alone(self):
        assert Gemma._try_sanitize_output("Art Memes Nature") == "Art Memes Nature"


class TestValidateResponse:
    def test_accepts_exactly_top_k_distinct_known_categories(self, gemma):
        assert gemma._validate_response("Art Memes Nature")

    def test_rejects_a_category_outside_the_list(self, gemma):
        assert not gemma._validate_response("Art Memes Sandwiches")

    def test_rejects_too_few_categories(self, gemma):
        assert not gemma._validate_response("Art Memes")

    def test_rejects_too_many_categories(self, gemma):
        assert not gemma._validate_response("Art Memes Nature People")

    def test_rejects_prose(self, gemma):
        assert not gemma._validate_response("This image looks like Art to me")

    def test_honors_an_explicit_top_k_override(self, gemma):
        assert gemma._validate_response("Art Memes", top_k=2)


class TestCreatePrompt:
    def test_includes_the_image_and_every_category(self, gemma, image_paths):
        prompt = gemma.create_prompt(image_paths[0])
        user_content = prompt[-1]["content"]

        image_part = next(part for part in user_content if part["type"] == "image")
        assert image_part["image"].size == (gemma.image_size, gemma.image_size)

        text = next(part for part in user_content if part["type"] == "text")["text"]
        assert all(category in text for category in gemma.categories)
        assert "<category> <category> <category>" in text

    def test_overrides_flow_into_the_requested_format(self, gemma, image_paths):
        prompt = gemma.create_prompt(image_paths[0], categories=["A", "B"], top_k=2)
        text = next(part for part in prompt[-1]["content"] if part["type"] == "text")[
            "text"
        ]
        assert "<category> <category>\n" in text
        assert "best 2 classifications" in text
