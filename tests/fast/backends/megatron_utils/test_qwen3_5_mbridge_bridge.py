import json
from types import SimpleNamespace

import torch
from safetensors.torch import save_file

from miles_plugins.mbridge.qwen3_5_moe import (
    Qwen3_5MoeBridge,
    _Qwen3_5ExpertSafeTensorIO,
)


def test_qwen3_5_expert_safetensor_io_loads_virtual_expert_slice(tmp_path):
    base_name = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    expert_tensor = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)

    save_file({base_name: expert_tensor}, tmp_path / "model.safetensors")
    index = {
        "metadata": {"total_size": expert_tensor.numel() * expert_tensor.element_size()},
        "weight_map": {base_name: "model.safetensors"},
    }
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps(index))

    tensor_io = _Qwen3_5ExpertSafeTensorIO(str(tmp_path), max_cached_weights=1)
    weight_name_0 = tensor_io.make_expert_weight_name(base_name, 0)
    weight_name_1 = tensor_io.make_expert_weight_name(base_name, 1)

    loaded = tensor_io.load_some_hf_weight([weight_name_0, weight_name_1])

    assert torch.equal(loaded[weight_name_0], expert_tensor[0])
    assert torch.equal(loaded[weight_name_1], expert_tensor[1])


def test_qwen3_5_bridge_maps_expert_weights_to_virtual_names():
    bridge = object.__new__(Qwen3_5MoeBridge)

    assert bridge._weight_name_mapping_mlp("decoder.layers.3.mlp.experts.linear_fc1.weight7") == [
        "model.language_model.layers.3.mlp.experts.gate_up_proj#expert7"
    ]
    assert bridge._weight_name_mapping_mlp("decoder.layers.3.mlp.experts.linear_fc2.weight7") == [
        "model.language_model.layers.3.mlp.experts.down_proj#expert7"
    ]


def test_qwen3_5_bridge_reaggregates_experts_on_export():
    bridge = object.__new__(Qwen3_5MoeBridge)
    bridge.hf_config = SimpleNamespace(text_config=SimpleNamespace(num_experts=2))

    first = bridge._weight_to_hf_format(
        "decoder.layers.0.mlp.experts.linear_fc1.weight0",
        torch.full((3, 4), 1.0),
    )
    second = bridge._weight_to_hf_format(
        "decoder.layers.0.mlp.experts.linear_fc1.weight1",
        torch.full((3, 4), 2.0),
    )

    assert first == ([], [])
    assert second[0] == ["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    assert torch.equal(
        second[1][0],
        torch.stack([torch.full((3, 4), 1.0), torch.full((3, 4), 2.0)], dim=0),
    )
