import json

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from miles.utils.hf_checkpoint_utils import HfSafetensorShardWriter, merge_missing_hf_tensors


def test_merge_missing_hf_tensors_restores_missing_weights(tmp_path):
    origin_dir = tmp_path / "origin"
    output_dir = tmp_path / "output"
    origin_dir.mkdir()
    output_dir.mkdir()

    text_tensor = torch.arange(8, dtype=torch.float32).reshape(2, 4)
    visual_tensor = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    save_file({"model.language_model.embed_tokens.weight": text_tensor}, origin_dir / "model-00001-of-00002.safetensors")
    save_file({"model.visual.patch_embed.proj.weight": visual_tensor}, origin_dir / "model-00002-of-00002.safetensors")
    (origin_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": text_tensor.numel() * text_tensor.element_size() + visual_tensor.numel() * visual_tensor.element_size()},
                "weight_map": {
                    "model.language_model.embed_tokens.weight": "model-00001-of-00002.safetensors",
                    "model.visual.patch_embed.proj.weight": "model-00002-of-00002.safetensors",
                },
            },
            indent=2,
        )
    )

    save_file({"model.language_model.embed_tokens.weight": text_tensor}, output_dir / "model-00001-of-00001.safetensors")
    (output_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": text_tensor.numel() * text_tensor.element_size()},
                "weight_map": {
                    "model.language_model.embed_tokens.weight": "model-00001-of-00001.safetensors",
                },
            },
            indent=2,
        )
    )

    missing_keys = merge_missing_hf_tensors(origin_dir, output_dir, chunk_size=1024)

    assert missing_keys == ["model.visual.patch_embed.proj.weight"]

    merged_index = json.loads((output_dir / "model.safetensors.index.json").read_text())
    passthrough_name = merged_index["weight_map"]["model.visual.patch_embed.proj.weight"]
    assert passthrough_name.startswith("model-passthrough-")
    with safe_open(output_dir / passthrough_name, framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor("model.visual.patch_embed.proj.weight"), visual_tensor)


def test_hf_safetensor_shard_writer_writes_index_and_shards(tmp_path):
    writer = HfSafetensorShardWriter(tmp_path)
    tensor_a = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    tensor_b = torch.arange(4, dtype=torch.float32).reshape(2, 2)

    first_shard = writer.add_chunk([("model.a.weight", tensor_a)])
    second_shard = writer.add_chunk([("model.b.weight", torch.nn.Parameter(tensor_b))])
    num_shards = writer.finalize()

    assert first_shard == "model-00000.safetensors"
    assert second_shard == "model-00001.safetensors"
    assert num_shards == 2

    index = json.loads((tmp_path / "model.safetensors.index.json").read_text())
    assert index["weight_map"]["model.a.weight"] == first_shard
    assert index["weight_map"]["model.b.weight"] == second_shard

    with safe_open(tmp_path / first_shard, framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor("model.a.weight"), tensor_a)
    with safe_open(tmp_path / second_shard, framework="pt", device="cpu") as handle:
        assert torch.equal(handle.get_tensor("model.b.weight"), tensor_b)
