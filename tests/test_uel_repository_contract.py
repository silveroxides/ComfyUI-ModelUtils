import importlib.util
from pathlib import Path

import pytest
import torch
from unifiedefficientloader import IncrementalSafetensorsWriter


REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_uel_io():
    spec = importlib.util.spec_from_file_location(
        "modelutils_uel_io_test", REPO_ROOT / "nodes" / "uel_io.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Handler:
    def __init__(self, values):
        self.values = values
        self.calls = []
        self.processed = []
        self.closed = 0

    def async_stream(self, keys, **kwargs):
        self.calls.append((list(keys), kwargs))

        def generate():
            try:
                for key in keys:
                    yield [(key, self.values[key])]
            finally:
                self.closed += 1

        return generate()

    def mark_processed(self, key):
        self.processed.append(key)


def test_production_nodes_have_no_synchronous_safetensors_io():
    violations = []
    for path in sorted((REPO_ROOT / "nodes").glob("*.py")):
        source = path.read_text(encoding="utf-8")
        if ".get_tensor(" in source:
            violations.append(f"{path.name}: get_tensor")
        if "from safetensors" in source or "import safetensors" in source:
            violations.append(f"{path.name}: direct safetensors import")
    assert violations == []


def test_work_units_use_bounded_streams_and_release_every_yield():
    uel = _load_uel_io()
    first = _Handler({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
    second = _Handler({"x": torch.tensor([3.0])})
    units = [("one", {0: ["a"], 1: ["x"]}), ("two", {0: ["b"]})]

    observed = []
    for logical, loaded in uel.stream_work_units(
        {0: first, 1: second}, units, pin_memory=True
    ):
        observed.append((logical, sorted(loaded)))

    assert observed == [
        ("one", [(0, "a"), (1, "x")]),
        ("two", [(0, "b")]),
    ]
    assert first.processed == ["a", "b"]
    assert second.processed == ["x"]
    assert first.closed == second.closed == 1
    for handler in (first, second):
        assert handler.calls[0][1] == {
            "batch_size": 1,
            "prefetch_batches": 1,
            "pin_memory": True,
        }


def test_work_unit_failure_releases_consumed_sources_and_closes_streams():
    uel = _load_uel_io()
    handler = _Handler({"a": torch.tensor([1.0]), "b": torch.tensor([2.0])})
    stream = uel.stream_work_units(
        {0: handler}, [("one", {0: ["a"]}), ("two", {0: ["b"]})],
        pin_memory=False,
    )

    with pytest.raises(RuntimeError, match="consumer failed"):
        try:
            for _, _ in stream:
                raise RuntimeError("consumer failed")
        finally:
            stream.close()

    assert handler.processed == ["a"]
    assert handler.closed == 1


def test_atomic_writer_preserves_destination_on_failure(tmp_path):
    uel = _load_uel_io()
    output = tmp_path / "output.safetensors"
    with IncrementalSafetensorsWriter(str(output), max_workers=1) as writer:
        writer.write("old", torch.tensor([1.0]))
    original = output.read_bytes()

    with pytest.raises(RuntimeError, match="planned failure"):
        with uel.atomic_uel_writer(str(output)) as writer:
            writer.write("new", torch.tensor([2.0]))
            raise RuntimeError("planned failure")

    assert output.read_bytes() == original
    assert list(tmp_path.glob(".output.safetensors.*.tmp")) == []
