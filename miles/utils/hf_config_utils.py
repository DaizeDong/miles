import json
from functools import lru_cache
from pathlib import Path

from transformers import AutoConfig


@lru_cache(maxsize=1)
def maybe_register_sglang_hf_configs() -> None:
    """Register custom SGLang config classes with transformers.AutoConfig when available."""
    try:
        import sglang.srt.utils.hf_transformers_utils  # noqa: F401
    except Exception:
        # Keep default transformers behavior when sglang is unavailable.
        return


def load_hf_config(hf_checkpoint: str, *, trust_remote_code: bool = True):
    maybe_register_sglang_hf_configs()
    return AutoConfig.from_pretrained(hf_checkpoint, trust_remote_code=trust_remote_code)


def get_hf_model_type(hf_checkpoint: str) -> str | None:
    try:
        return getattr(load_hf_config(hf_checkpoint, trust_remote_code=True), "model_type", None)
    except Exception:
        return load_hf_config_json(hf_checkpoint).get("model_type")


def load_hf_config_json(hf_checkpoint: str) -> dict:
    config_path = Path(hf_checkpoint) / "config.json"
    if not config_path.exists():
        return {}
    return json.loads(config_path.read_text(encoding="utf-8"))


def get_hf_text_config_dict(hf_checkpoint: str) -> dict:
    config_dict = load_hf_config_json(hf_checkpoint)
    return config_dict.get("text_config", config_dict)
