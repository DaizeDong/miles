"""Predictive-routing bridge patch for qwen3_moe.

Miles only needs one extension over the upstream mbridge qwen3_moe bridge:
export/load support for ``mlp.router.bias_predictor.weight``.
"""

from mbridge.models import Qwen2MoEBridge

try:
    from mbridge.models import Qwen3MoeBridge as _BaseQwen3MoeBridge
except ImportError:
    try:
        from mbridge.models import Qwen3MoEBridge as _BaseQwen3MoeBridge
    except ImportError:
        _BaseQwen3MoeBridge = Qwen2MoEBridge


def _patch_qwen3_moe_bridge():
    predictor_mapping = {
        "mlp.router.bias_predictor.weight": [
            "model.layers.{layer_number}.mlp.bias_predictor.weight",
        ]
    }
    bridge_cls = _BaseQwen3MoeBridge
    if getattr(bridge_cls, "_miles_predictive_patch_applied", False):
        return bridge_cls

    bridge_cls._MLP_MAPPING = {**getattr(bridge_cls, "_MLP_MAPPING", {}), **predictor_mapping}

    original_weight_name_mapping_mlp = bridge_cls._weight_name_mapping_mlp

    def _weight_name_mapping_mlp(self, name: str) -> list[str]:
        if "mlp.router.bias_predictor.weight" in name:
            layer_number = name.split(".")[2]
            return [x.format(layer_number=layer_number) for x in predictor_mapping["mlp.router.bias_predictor.weight"]]
        return original_weight_name_mapping_mlp(self, name)

    bridge_cls._weight_name_mapping_mlp = _weight_name_mapping_mlp
    bridge_cls._miles_predictive_patch_applied = True
    return bridge_cls


Qwen3MoePredictiveBridge = _patch_qwen3_moe_bridge()

