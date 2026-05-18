from types import SimpleNamespace

from miles.rollout.generate_utils.generate_endpoint_utils import compute_prompt_ids_from_sample
from miles.utils.types import Sample


class _DummyProcessor:
    def __call__(self, text, **kwargs):
        assert kwargs == {}
        return {
            "input_ids": [[11, 12, 13]],
            "attention_mask": [[1, 1, 1]],
            "pixel_values": "vision-features",
        }


def test_compute_prompt_ids_from_sample_supports_text_only_sample_with_processor():
    state = SimpleNamespace(processor=_DummyProcessor(), tokenizer=None)
    sample = Sample(prompt="solve", multimodal_inputs=None)

    prompt_ids = compute_prompt_ids_from_sample(state, sample)

    assert prompt_ids == [11, 12, 13]
    assert sample.multimodal_train_inputs == {"pixel_values": "vision-features"}
