"""Regression tests for the SGLang transactional weight-update protocol."""

from argparse import Namespace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from miles.backends.megatron_utils.update_weight import update_weight_from_tensor as update_weight_module
from miles.backends.megatron_utils.update_weight.update_weight_from_tensor import (
    UpdateWeightFromTensor,
    _run_weight_update_session_phase,
)
from miles.backends.sglang_utils.sglang_engine import SGLangEngine


_MODULE = "miles.backends.megatron_utils.update_weight.update_weight_from_tensor"


class _RemoteMethod:
    def __init__(self, events, phase, engine_index):
        self.events = events
        self.phase = phase
        self.engine_index = engine_index

    def remote(self, *args, **kwargs):
        self.events.append((self.phase, self.engine_index, args, kwargs))
        return (self.phase, self.engine_index)


class _Engine:
    def __init__(self, events, engine_index):
        for phase in (
            "pause",
            "flush",
            "begin",
            "end",
            "continue",
        ):
            method_name = {
                "pause": "pause_generation",
                "flush": "flush_cache",
                "begin": "begin_weight_update",
                "end": "end_weight_update",
                "continue": "continue_generation",
            }[phase]
            setattr(self, method_name, _RemoteMethod(events, phase, engine_index))


def _phase_indices(events, phase):
    return [index for index, event in enumerate(events) if event[0] == phase]


def _updater(events, chunks=3):
    updater = object.__new__(UpdateWeightFromTensor)
    updater.args = Namespace(pause_generation_mode="retract")
    updater.weight_version = 0
    updater.is_lora = False
    updater.use_distribute = False
    # Exercise the branch that previously called the retired
    # /post_process_weights endpoint for compressed-tensors models.
    updater.quantization_config = {"quant_method": "compressed-tensors"}
    updater.rollout_engines = [_Engine(events, index) for index in range(8)]
    updater.weights_getter = lambda: {"weight": object()}
    updater._hf_weight_iterator = MagicMock()
    updater._hf_weight_iterator.get_hf_weight_chunks.return_value = iter(
        [[(f"weight-{index}", object())] for index in range(chunks)]
    )

    chunk_counter = iter(range(chunks))

    def send_base(_chunk):
        index = next(chunk_counter)
        events.append(("chunk", index, (), {}))
        return [("chunk-result", index)], [object()]

    updater._send_base_params = send_base
    return updater


def _fake_ray_get(refs, *, failed_begin_index=None):
    if not isinstance(refs, list):
        return refs
    results = []
    for ref in refs:
        if not isinstance(ref, tuple):
            results.append(None)
        elif ref[0] == "begin":
            if ref[1] == failed_begin_index:
                results.append({"success": False, "message": "session rejected"})
            else:
                results.append({"success": True, "message": "Success"})
        elif ref[0] == "end":
            results.append({"success": True, "message": "Success"})
        elif ref[0] == "chunk-result":
            results.append({"success": True, "message": "Success"})
        else:
            results.append(None)
    return results


class TestSGLangEngineSessionEndpoints:
    def test_engine_uses_exact_begin_and_end_endpoints(self):
        engine = object.__new__(SGLangEngine)
        engine._make_request = MagicMock(side_effect=[{"success": True}, {"success": True}])

        assert engine.begin_weight_update(selector="all") == {"success": True}
        assert engine.end_weight_update() == {"success": True}
        assert engine._make_request.call_args_list[0].args == (
            "begin_weight_update",
            {"selector": "all"},
        )
        assert engine._make_request.call_args_list[1].args == (
            "end_weight_update",
            {},
        )


class TestTransactionalWeightUpdate:
    @patch(f"{_MODULE}.get_gloo_group", return_value=object())
    @patch(f"{_MODULE}.dist")
    @patch(f"{_MODULE}.ray")
    def test_eight_engines_enclose_all_tensor_chunks_before_resume(
        self,
        mock_ray,
        mock_dist,
        _mock_gloo,
    ):
        events = []
        updater = _updater(events, chunks=3)
        mock_dist.get_rank.return_value = 0
        mock_ray.get.side_effect = _fake_ray_get

        updater.update_weights()

        assert len(_phase_indices(events, "begin")) == 8
        assert len(_phase_indices(events, "chunk")) == 3
        assert len(_phase_indices(events, "end")) == 8
        assert len(_phase_indices(events, "continue")) == 8
        assert max(_phase_indices(events, "begin")) < min(_phase_indices(events, "chunk"))
        assert max(_phase_indices(events, "chunk")) < min(_phase_indices(events, "end"))
        assert max(_phase_indices(events, "end")) < min(_phase_indices(events, "continue"))
        assert all(event[3] == {"selector": "all"} for event in events if event[0] == "begin")

    @patch(f"{_MODULE}.get_gloo_group", return_value=object())
    @patch(f"{_MODULE}.dist")
    @patch(f"{_MODULE}.ray")
    def test_rejected_begin_fails_before_chunks_end_or_resume(
        self,
        mock_ray,
        mock_dist,
        _mock_gloo,
    ):
        events = []
        updater = _updater(events, chunks=3)
        mock_dist.get_rank.return_value = 0
        mock_ray.get.side_effect = lambda refs: _fake_ray_get(refs, failed_begin_index=4)

        with pytest.raises(RuntimeError, match=r"begin_weight_update failed.*engine 4.*session rejected"):
            updater.update_weights()

        assert len(_phase_indices(events, "begin")) == 8
        assert not _phase_indices(events, "chunk")
        assert not _phase_indices(events, "end")
        assert not _phase_indices(events, "continue")

    @patch(f"{_MODULE}.ray")
    def test_session_result_cardinality_is_fail_closed(self, mock_ray):
        engines = [MagicMock(), MagicMock()]
        mock_ray.get.return_value = [{"success": True}]
        with pytest.raises(RuntimeError, match="returned 1 results for 2 engines"):
            _run_weight_update_session_phase(engines, phase="begin")

    def test_unknown_session_phase_is_rejected(self):
        with pytest.raises(ValueError, match="unknown weight-update session phase"):
            _run_weight_update_session_phase([], phase="commit")

    def test_tensor_orchestrator_does_not_call_legacy_post_process(self):
        source = Path(update_weight_module.__file__).read_text(encoding="utf-8")
        function = source[
            source.index("    def update_weights(self) -> None:") : source.index("    def _send_base_params")
        ]
        assert "post_process_weights" not in function
        assert function.index('phase="begin"') < function.index("get_hf_weight_chunks")
        assert function.rindex("get_hf_weight_chunks") < function.index('phase="end"')
        assert function.index('phase="end"') < function.index("continue_generation")
