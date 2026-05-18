from miles.utils.data import filter_long_prompt
from miles.utils.processing_utils import prompt_has_vision_inputs
from miles.utils.types import Sample


class _DummyProcessor:
    def __call__(self, text, **kwargs):
        return {"input_ids": [list(range(len(text)))]}


def test_prompt_has_vision_inputs_returns_false_for_text_only_messages():
    prompt = [{"role": "user", "content": "Solve x^2=1."}]
    assert prompt_has_vision_inputs(prompt) is False


def test_prompt_has_vision_inputs_returns_true_for_image_messages():
    prompt = [{"role": "user", "content": [{"type": "image", "image": "a.png"}, {"type": "text", "text": "desc"}]}]
    assert prompt_has_vision_inputs(prompt) is True


def test_filter_long_prompt_supports_text_only_samples_with_processor():
    samples = [
        Sample(prompt="abcd", multimodal_inputs=None),
        Sample(prompt="abcdefgh", multimodal_inputs=None),
    ]

    filtered = filter_long_prompt(
        samples,
        tokenizer=None,
        processor=_DummyProcessor(),
        max_length=4,
    )

    assert [sample.prompt for sample in filtered] == ["abcd"]
