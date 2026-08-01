import ast
from collections.abc import Mapping, Sequence
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
UPDATER = ROOT / "miles/backends/megatron_utils/update_weight/update_weight_from_tensor.py"
ENGINE = ROOT / "miles/backends/sglang_utils/sglang_engine.py"


def _module_tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def _function(path: Path, name: str) -> ast.FunctionDef:
    for node in _module_tree(path).body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
        if isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, ast.FunctionDef) and child.name == name:
                    return child
    raise AssertionError(f"function {name!r} not found in {path}")


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
        self.begin_weight_update = _RemoteMethod(events, "begin", engine_index)
        self.end_weight_update = _RemoteMethod(events, "end", engine_index)


class _Ray:
    def __init__(
        self,
        *,
        failed_phase=None,
        failed_index=None,
        malformed_phase=None,
        malformed_index=None,
        malformed_value=None,
        truncate=False,
    ):
        self.failed_phase = failed_phase
        self.failed_index = failed_index
        self.malformed_phase = malformed_phase
        self.malformed_index = malformed_index
        self.malformed_value = malformed_value
        self.truncate = truncate

    def get(self, refs):
        results = []
        for phase, engine_index in refs:
            if phase == self.malformed_phase and engine_index == self.malformed_index:
                results.append(self.malformed_value)
            elif phase == self.failed_phase and engine_index == self.failed_index:
                results.append({"success": False, "message": "session rejected"})
            else:
                results.append({"success": True, "message": "Success"})
        return results[:-1] if self.truncate else results


def _load_phase_helper(ray):
    node = _function(UPDATER, "_run_weight_update_session_phase")
    module = ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[]))
    namespace = {
        "ray": ray,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "ActorHandle": object,
    }
    exec(compile(module, str(UPDATER), "exec"), namespace)
    return namespace["_run_weight_update_session_phase"]


class SGLangWeightSessionContractTests(unittest.TestCase):
    def test_eight_engines_enclose_three_chunks(self) -> None:
        events = []
        engines = [_Engine(events, index) for index in range(8)]
        helper = _load_phase_helper(_Ray())

        helper(engines, phase="begin")
        events.extend(("chunk", index, (), {}) for index in range(3))
        helper(engines, phase="end")

        phases = [event[0] for event in events]
        self.assertEqual(phases, ["begin"] * 8 + ["chunk"] * 3 + ["end"] * 8)
        self.assertTrue(
            all(event[3] == {"selector": "all"} for event in events if event[0] == "begin")
        )

    def test_failed_begin_response_is_rejected(self) -> None:
        events = []
        engines = [_Engine(events, index) for index in range(8)]
        helper = _load_phase_helper(_Ray(failed_phase="begin", failed_index=4))
        with self.assertRaisesRegex(
            RuntimeError,
            r"begin_weight_update failed on rollout engine 4: session rejected",
        ):
            helper(engines, phase="begin")

    def test_failed_end_response_is_rejected(self) -> None:
        events = []
        engines = [_Engine(events, index) for index in range(8)]
        helper = _load_phase_helper(_Ray(failed_phase="end", failed_index=6))
        with self.assertRaisesRegex(
            RuntimeError,
            r"end_weight_update failed on rollout engine 6: session rejected",
        ):
            helper(engines, phase="end")

    def test_none_and_malformed_responses_are_rejected(self) -> None:
        cases = ((None, "unexpected response type NoneType"), ("ok", "unexpected response type str"))
        for value, expected in cases:
            with self.subTest(value=value):
                events = []
                engines = [_Engine(events, index) for index in range(8)]
                helper = _load_phase_helper(
                    _Ray(
                        malformed_phase="begin",
                        malformed_index=2,
                        malformed_value=value,
                    )
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    helper(engines, phase="begin")

    def test_non_boolean_success_is_rejected(self) -> None:
        events = []
        engines = [_Engine(events, index) for index in range(8)]
        helper = _load_phase_helper(
            _Ray(
                malformed_phase="end",
                malformed_index=1,
                malformed_value={"success": "true", "message": "not a bool"},
            )
        )
        with self.assertRaisesRegex(RuntimeError, r"end_weight_update failed.*not a bool"):
            helper(engines, phase="end")

    def test_missing_engine_response_is_rejected(self) -> None:
        events = []
        engines = [_Engine(events, index) for index in range(8)]
        helper = _load_phase_helper(_Ray(truncate=True))
        with self.assertRaisesRegex(RuntimeError, r"returned 7 results for 8 engines"):
            helper(engines, phase="begin")

    def test_unknown_phase_is_rejected(self) -> None:
        helper = _load_phase_helper(_Ray())
        with self.assertRaisesRegex(ValueError, "unknown weight-update session phase"):
            helper([], phase="commit")

    def test_production_orchestrator_order_and_legacy_endpoint_removal(self) -> None:
        source = UPDATER.read_text(encoding="utf-8")
        node = _function(UPDATER, "update_weights")
        segment = ast.get_source_segment(source, node)
        self.assertIsNotNone(segment)
        assert segment is not None
        self.assertNotIn("post_process_weights", segment)
        self.assertLess(segment.index('phase="begin"'), segment.index("get_hf_weight_chunks"))
        self.assertLess(segment.rindex("get_hf_weight_chunks"), segment.index('phase="end"'))
        self.assertLess(segment.index('phase="end"'), segment.index("continue_generation"))

    def test_skip_base_sync_guards_both_session_phases(self) -> None:
        source = UPDATER.read_text(encoding="utf-8")
        node = _function(UPDATER, "update_weights")
        guarded_phases = []
        for candidate in ast.walk(node):
            if not isinstance(candidate, ast.If):
                continue
            if ast.unparse(candidate.test) != "not skip_base_sync":
                continue
            guarded_phases.extend(
                keyword.value.value
                for child in ast.walk(candidate)
                if isinstance(child, ast.Call)
                and isinstance(child.func, ast.Name)
                and child.func.id == "_run_weight_update_session_phase"
                for keyword in child.keywords
                if keyword.arg == "phase" and isinstance(keyword.value, ast.Constant)
            )
        self.assertCountEqual(guarded_phases, ["begin", "end"])

    def test_transaction_scope_explicitly_excludes_distributed_engines(self) -> None:
        node = _function(UPDATER, "update_weights")
        calls = [
            child
            for child in ast.walk(node)
            if isinstance(child, ast.Call)
            and isinstance(child.func, ast.Name)
            and child.func.id == "_run_weight_update_session_phase"
        ]
        self.assertEqual(len(calls), 2)
        self.assertTrue(all(ast.unparse(call.args[0]) == "self.rollout_engines" for call in calls))
        self.assertTrue(all("distributed_rollout_engines" not in ast.unparse(call) for call in calls))

    def test_engine_methods_target_exact_transaction_endpoints(self) -> None:
        source = ENGINE.read_text(encoding="utf-8")
        begin = ast.get_source_segment(source, _function(ENGINE, "begin_weight_update"))
        end = ast.get_source_segment(source, _function(ENGINE, "end_weight_update"))
        self.assertIsNotNone(begin)
        self.assertIsNotNone(end)
        assert begin is not None and end is not None
        self.assertIn('"begin_weight_update"', begin)
        self.assertIn('{"selector": selector}', begin)
        self.assertIn('"end_weight_update"', end)
        self.assertIn("{}", end)


if __name__ == "__main__":
    unittest.main()
