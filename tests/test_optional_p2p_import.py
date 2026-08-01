import ast
import builtins
import copy
from pathlib import Path
from types import SimpleNamespace
import textwrap
import unittest
from unittest import mock


ACTOR = (
    Path(__file__).resolve().parents[1]
    / "miles/backends/megatron_utils/actor.py"
)
EXPECTED_SELECTION = """if self.args.colocate:
    update_weight_cls = UpdateWeightFromTensor
else:
    if self.args.update_weight_transfer_mode == \"broadcast\":
        update_weight_cls = UpdateWeightFromDistributed
    else:
        if UpdateWeightP2P is None:
            raise RuntimeError(
                \"UpdateWeightP2P is not importable in this environment \"
                f\"(SIF sglang version mismatch): {_update_weight_p2p_import_error}\"
            )
        update_weight_cls = UpdateWeightP2P"""


def _actor_tree() -> ast.Module:
    return ast.parse(ACTOR.read_text(encoding="utf-8"))


def _p2p_import_guard() -> ast.Try:
    for node in ast.walk(_actor_tree()):
        if not isinstance(node, ast.Try):
            continue
        imports = [item for item in ast.walk(node) if isinstance(item, ast.ImportFrom)]
        if any(item.module and item.module.endswith("update_weight_from_distributed.p2p") for item in imports):
            return copy.deepcopy(node)
    raise AssertionError("optional P2P import guard not found")


def _selection_block() -> ast.If:
    for node in ast.walk(_actor_tree()):
        if isinstance(node, ast.If) and ast.unparse(node.test) == "self.args.colocate":
            return copy.deepcopy(node)
    raise AssertionError("production update-weight selection block not found")


def _execute_guard(importer):
    namespace = {"__name__": "p2p_guard_probe", "__package__": "miles.backends.megatron_utils"}
    module = ast.fix_missing_locations(ast.Module(body=[_p2p_import_guard()], type_ignores=[]))
    with mock.patch("builtins.__import__", side_effect=importer):
        exec(compile(module, str(ACTOR), "exec"), namespace)
    return namespace


def _execute_selection(*, colocate: bool, mode: str, p2p, import_error=None):
    tensor_cls = object()
    distributed_cls = object()
    namespace = {
        "self": SimpleNamespace(args=SimpleNamespace(colocate=colocate, update_weight_transfer_mode=mode)),
        "UpdateWeightFromTensor": tensor_cls,
        "UpdateWeightFromDistributed": distributed_cls,
        "UpdateWeightP2P": p2p,
        "_update_weight_p2p_import_error": import_error,
    }
    module = ast.fix_missing_locations(ast.Module(body=[_selection_block()], type_ignores=[]))
    exec(compile(module, str(ACTOR), "exec"), namespace)
    return namespace["update_weight_cls"], tensor_cls, distributed_cls


class OptionalP2PImportTests(unittest.TestCase):
    def test_missing_mooncake_is_the_only_suppressed_import(self) -> None:
        error = ModuleNotFoundError("No module named 'mooncake'", name="mooncake")

        def importer(*_args, **_kwargs):
            raise error

        namespace = _execute_guard(importer)
        self.assertIsNone(namespace["UpdateWeightP2P"])
        self.assertIs(namespace["_update_weight_p2p_import_error"], error)

    def test_unrelated_missing_module_is_reraised_unchanged(self) -> None:
        error = ModuleNotFoundError("No module named 'unrelated_dependency'", name="unrelated_dependency")

        def importer(*_args, **_kwargs):
            raise error

        with self.assertRaises(ModuleNotFoundError) as raised:
            _execute_guard(importer)
        self.assertIs(raised.exception, error)

    def test_successful_p2p_import_sets_none_error(self) -> None:
        sentinel = object()

        def importer(*_args, **_kwargs):
            return SimpleNamespace(UpdateWeightP2P=sentinel)

        namespace = _execute_guard(importer)
        self.assertIs(namespace["UpdateWeightP2P"], sentinel)
        self.assertIsNone(namespace["_update_weight_p2p_import_error"])

    def test_original_selection_block_is_byte_for_byte_unchanged(self) -> None:
        source = ACTOR.read_text(encoding="utf-8")
        node = _selection_block()
        segment = "\n".join(source.splitlines()[node.lineno - 1 : node.end_lineno])
        self.assertEqual(segment, textwrap.indent(EXPECTED_SELECTION, " " * node.col_offset))

    def test_active_colocated_mode_selects_tensor_updater(self) -> None:
        selected, tensor_cls, _distributed_cls = _execute_selection(
            colocate=True, mode="broadcast", p2p=None
        )
        self.assertIs(selected, tensor_cls)

    def test_noncolocated_broadcast_selects_distributed_updater(self) -> None:
        selected, _tensor_cls, distributed_cls = _execute_selection(
            colocate=False, mode="broadcast", p2p=None
        )
        self.assertIs(selected, distributed_cls)

    def test_explicit_p2p_success_and_missing_dependency_fail_closed(self) -> None:
        sentinel = object()
        selected, _tensor_cls, _distributed_cls = _execute_selection(
            colocate=False, mode="p2p", p2p=sentinel
        )
        self.assertIs(selected, sentinel)
        error = ModuleNotFoundError("No module named 'mooncake'", name="mooncake")
        with self.assertRaisesRegex(RuntimeError, "UpdateWeightP2P is not importable"):
            _execute_selection(colocate=False, mode="p2p", p2p=None, import_error=error)


if __name__ == "__main__":
    unittest.main()
