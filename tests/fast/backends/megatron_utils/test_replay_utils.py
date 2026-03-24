import importlib
import sys
import types
from types import SimpleNamespace

import torch
import torch.nn as nn


def _install_megatron_replay_stubs():
    if "megatron.core.transformer.transformer_layer" in sys.modules:
        return

    megatron_module = types.ModuleType("megatron")
    core_module = types.ModuleType("megatron.core")
    transformer_module = types.ModuleType("megatron.core.transformer")
    transformer_layer_module = types.ModuleType("megatron.core.transformer.transformer_layer")
    transformer_layer_module.get_transformer_layer_offset = lambda config, vp_stage=0: 0

    transformer_module.transformer_layer = transformer_layer_module
    core_module.transformer = transformer_module
    megatron_module.core = core_module

    sys.modules["megatron"] = megatron_module
    sys.modules["megatron.core"] = core_module
    sys.modules["megatron.core.transformer"] = transformer_module
    sys.modules["megatron.core.transformer.transformer_layer"] = transformer_layer_module


_install_megatron_replay_stubs()
replay_utils = importlib.import_module("miles.backends.megatron_utils.replay_utils")


class _DummyReplay:
    def __init__(self):
        self.records = []

    def record(self, value):
        self.records.append(value.clone())


class _DummyRouter(nn.Module):
    def __init__(self, replay):
        super().__init__()
        self.routing_replay = replay


class _DummyLayer(nn.Module):
    def __init__(self, replay=None):
        super().__init__()
        if replay is not None:
            self.router = _DummyRouter(replay)


class _DummyModel(nn.Module):
    def __init__(self, replays_by_local_layer):
        super().__init__()
        self.config = SimpleNamespace()
        self.decoder = nn.Module()
        max_local_layer = max(replays_by_local_layer) if replays_by_local_layer else -1
        self.decoder.layers = nn.ModuleList(
            [_DummyLayer(replays_by_local_layer.get(local_layer_idx)) for local_layer_idx in range(max_local_layer + 1)]
        )


def test_register_replay_list_moe_uses_actual_local_router_modules(monkeypatch):
    monkeypatch.setattr(replay_utils, "get_transformer_layer_offset", lambda config, vp_stage=0: 16 * vp_stage + 16)

    replay0 = _DummyReplay()
    replay2 = _DummyReplay()
    model = SimpleNamespace(module=_DummyModel({0: replay0, 2: replay2}))

    replay_data = torch.arange(3 * 20 * 4, dtype=torch.int64).reshape(3, 20, 4)

    replay_utils._register_replay_list_moe(replay_list=[], replay_data=replay_data, models=[model])

    assert len(replay0.records) == 1
    assert len(replay2.records) == 1
    assert torch.equal(replay0.records[0], replay_data[:, 16])
    assert torch.equal(replay2.records[0], replay_data[:, 18])


def test_register_replay_list_moe_rejects_unexpected_router_module_paths(monkeypatch):
    monkeypatch.setattr(replay_utils, "get_transformer_layer_offset", lambda config, vp_stage=0: 0)

    class _BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace()
            self.router = _DummyRouter(_DummyReplay())

    model = SimpleNamespace(module=_BadModel())
    replay_data = torch.zeros(2, 4, 8, dtype=torch.int64)

    try:
        replay_utils._register_replay_list_moe(replay_list=[], replay_data=replay_data, models=[model])
    except ValueError as exc:
        assert "unexpected module path" in str(exc)
    else:
        raise AssertionError("Expected _register_replay_list_moe to reject non-decoder router module paths.")
