import json
import shutil
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

DEFAULT_SAFETENSOR_CHUNK_SIZE = 5 * 1024**3


def copy_hf_non_weight_assets(origin_hf_dir, output_dir):
    origin_hf_dir = Path(origin_hf_dir)
    output_dir = Path(output_dir)
    for origin_path in origin_hf_dir.iterdir():
        if origin_path.name == "model.safetensors.index.json" or origin_path.suffix == ".safetensors":
            continue
        if not origin_path.is_file():
            continue
        shutil.copy(origin_path, output_dir / origin_path.name)


def merge_missing_hf_tensors(origin_hf_dir, output_dir, chunk_size=DEFAULT_SAFETENSOR_CHUNK_SIZE):
    origin_hf_dir = Path(origin_hf_dir)
    output_dir = Path(output_dir)

    origin_index = _load_weight_index(origin_hf_dir)
    output_index = _load_weight_index(output_dir)
    missing_keys = [key for key in origin_index["weight_map"] if key not in output_index["weight_map"]]
    if not missing_keys:
        return []

    shard_to_keys = defaultdict(list)
    for key in missing_keys:
        shard_to_keys[origin_index["weight_map"][key]].append(key)

    passthrough_shards = []
    current_tensors = {}
    current_size = 0
    total_added_size = 0

    def flush_current():
        nonlocal current_tensors, current_size
        if current_tensors:
            passthrough_shards.append(current_tensors)
            current_tensors = {}
            current_size = 0

    for shard_name, keys in shard_to_keys.items():
        shard_path = origin_hf_dir / shard_name
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in keys:
                tensor = handle.get_tensor(key)
                tensor_size = tensor.numel() * tensor.element_size()
                total_added_size += tensor_size
                if current_tensors and current_size + tensor_size > chunk_size:
                    flush_current()
                current_tensors[key] = tensor
                current_size += tensor_size
    flush_current()

    num_new_shards = len(passthrough_shards)
    for shard_idx, tensors in enumerate(passthrough_shards, start=1):
        filename = f"model-passthrough-{shard_idx:05d}-of-{num_new_shards:05d}.safetensors"
        save_file(tensors, output_dir / filename)
        for key in tensors:
            output_index["weight_map"][key] = filename

    output_index.setdefault("metadata", {})
    output_index["metadata"]["total_size"] = output_index["metadata"].get("total_size", 0) + total_added_size
    (output_dir / "model.safetensors.index.json").write_text(json.dumps(output_index, indent=2))
    return missing_keys


class HfSafetensorShardWriter:
    def __init__(self, output_dir):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._weight_map = {}
        self._total_size = 0
        self._num_shards = 0

    def add_chunk(self, named_tensors):
        named_tensors = list(named_tensors)
        if not named_tensors:
            return None

        shard_name = f"model-{self._num_shards:05d}.safetensors"
        shard_tensors = {}
        for key, tensor in named_tensors:
            cpu_tensor = self._to_cpu_tensor(tensor)
            if key in self._weight_map:
                raise ValueError(f"Duplicate HF tensor key during export: {key}")
            shard_tensors[key] = cpu_tensor
            self._weight_map[key] = shard_name
            self._total_size += cpu_tensor.numel() * cpu_tensor.element_size()

        save_file(shard_tensors, self.output_dir / shard_name)
        self._num_shards += 1
        return shard_name

    def finalize(self):
        if self._num_shards == 0:
            raise RuntimeError(f"No HF tensor shards were written under {self.output_dir}")

        index = {
            "metadata": {"total_size": self._total_size},
            "weight_map": self._weight_map,
        }
        (self.output_dir / "model.safetensors.index.json").write_text(json.dumps(index, indent=2))
        return self._num_shards

    @staticmethod
    def _to_cpu_tensor(tensor):
        if isinstance(tensor, torch.nn.Parameter):
            tensor = tensor.data
        return tensor.detach().cpu().contiguous()


def _load_weight_index(model_dir: Path):
    index_path = model_dir / "model.safetensors.index.json"
    if index_path.exists():
        return json.loads(index_path.read_text())

    safetensor_files = sorted(model_dir.glob("*.safetensors"))
    if not safetensor_files:
        raise FileNotFoundError(f"No safetensors weights found under {model_dir}")

    weight_map = {}
    total_size = 0
    for shard_path in safetensor_files:
        with safe_open(shard_path, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                weight_map[key] = shard_path.name
        total_size += shard_path.stat().st_size

    return {"metadata": {"total_size": total_size}, "weight_map": weight_map}
