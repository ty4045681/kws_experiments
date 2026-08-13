from __future__ import annotations

import contextlib
import copy
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "scripts" / "bench_zipformer_streaming_mindir_fixtures.py"


class FakeTensor:
    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        dtype: str,
        values: Optional[np.ndarray] = None,
    ) -> None:
        self.name = name
        self.shape = list(shape)
        self.dtype = dtype
        self.valid = True
        self.set_calls = 0
        if values is None:
            resolved = [value if value > 0 else 1 for value in self.shape]
            self.values = np.zeros(resolved, dtype=np.dtype(dtype))
        else:
            self.values = np.array(values, copy=True)

    def set_data_from_numpy(self, values: np.ndarray) -> None:
        if not self.valid:
            raise AssertionError("The benchmark reused a stale tensor handle after resize().")
        array = np.asarray(values)
        if list(array.shape) != self.shape:
            raise AssertionError(f"Unexpected data shape for {self.name}: {array.shape} != {self.shape}")
        self.values = np.array(array, copy=True)
        self.set_calls += 1

    def get_data_to_numpy(self) -> np.ndarray:
        if not self.valid:
            raise AssertionError("The benchmark read a stale tensor handle after resize().")
        return np.array(self.values, copy=True)


class FakeContext:
    def __init__(self) -> None:
        self.target: List[str] = []
        self.cpu = types.SimpleNamespace(
            thread_num=None,
            enable_fp16=None,
            precision_mode=None,
        )
        self.ascend = types.SimpleNamespace(
            device_id=None,
            precision_mode=None,
            provider=None,
        )


class FakeModel:
    def __init__(self, runtime: "FakeMindSporeLite") -> None:
        self.runtime = runtime
        self._inputs = [FakeTensor(*spec) for spec in runtime.input_specs]
        self.build_calls: List[Tuple[Tuple[Any, ...], Dict[str, Any]]] = []
        self.resize_calls: List[List[List[int]]] = []
        self.predict_snapshots: List[Dict[str, np.ndarray]] = []

    def build_from_file(self, *args: Any, **kwargs: Any) -> None:
        self.build_calls.append((args, kwargs))

    def get_inputs(self) -> List[FakeTensor]:
        return list(self._inputs)

    def resize(self, inputs: Sequence[FakeTensor], shapes: Sequence[Sequence[int]]) -> None:
        if list(inputs) != self._inputs:
            raise AssertionError("resize() did not receive the model's current input handles.")
        self.resize_calls.append([[int(value) for value in shape] for shape in shapes])
        old_inputs = self._inputs
        for tensor in old_inputs:
            tensor.valid = False
        self._inputs = [
            FakeTensor(old.name, shape, old.dtype)
            for old, shape in zip(old_inputs, shapes)
        ]

    def predict(self, inputs: Sequence[FakeTensor]) -> List[FakeTensor]:
        if list(inputs) != self._inputs:
            raise AssertionError("predict() did not receive refreshed model input handles.")
        snapshot = {
            tensor.name: tensor.get_data_to_numpy()
            for tensor in inputs
        }
        self.predict_snapshots.append(snapshot)
        features = snapshot["x"]
        cache = snapshot["cache"]
        processed_lens = snapshot["processed_lens"]
        encoder_out = np.mean(features, axis=1, keepdims=True).astype(np.float32)
        return [
            FakeTensor("encoder_out", encoder_out.shape, "float32", encoder_out),
            FakeTensor("new_cache", cache.shape, "float32", cache + 100.0),
            FakeTensor(
                "new_processed_lens",
                processed_lens.shape,
                "int64",
                processed_lens + 100,
            ),
        ]


class FakeMindSporeLite(types.ModuleType):
    def __init__(
        self,
        input_specs: Optional[Iterable[Tuple[str, Sequence[int], str]]] = None,
    ) -> None:
        super().__init__("mindspore_lite")
        self.__version__ = "fake-1.0"
        self.ModelType = types.SimpleNamespace(MINDIR="MINDIR")
        self.input_specs = list(
            input_specs
            or [
                ("x", [-1, 4, 2], "float32"),
                ("cache", [-1, 2], "float32"),
                ("processed_lens", [-1], "int64"),
            ]
        )
        self.contexts: List[FakeContext] = []
        self.models: List[FakeModel] = []
        self.Context = self._new_context
        self.Model = self._new_model

    def _new_context(self) -> FakeContext:
        context = FakeContext()
        self.contexts.append(context)
        return context

    def _new_model(self) -> FakeModel:
        model = FakeModel(self)
        self.models.append(model)
        return model


def load_benchmark(fake_mslite: FakeMindSporeLite) -> Any:
    if not BENCH_PATH.is_file():
        raise AssertionError(f"Benchmark script does not exist: {BENCH_PATH}")
    module_name = f"bench_zipformer_streaming_mindir_fixtures_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, BENCH_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {BENCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, {"mindspore_lite": fake_mslite}):
        spec.loader.exec_module(module)
    return module


def array_entry(index: int, name: str, array: np.ndarray, path: Path, root: Path) -> Dict[str, Any]:
    return {
        "index": index,
        "name": name,
        "dtype": str(array.dtype),
        "dtype_str": array.dtype.str,
        "shape": list(array.shape),
        "element_count": int(array.size),
        "byte_size": int(array.nbytes),
        "file": str(path.relative_to(root)),
    }


def make_fixtures(root: Path) -> Tuple[Path, Path, Dict[str, Any]]:
    fixtures_dir = root / "fixtures"
    fixtures_dir.mkdir()
    model_path = root / "encoder.mindir"
    model_path.write_bytes(b"fake mindir")
    step_values = [
        {
            "x": np.zeros((1, 4, 2), dtype=np.float32),
            "cache": np.array([[10.0, 11.0]], dtype=np.float32),
            "processed_lens": np.array([4], dtype=np.int64),
        },
        {
            "x": np.ones((1, 4, 2), dtype=np.float32),
            "cache": np.array([[20.0, 21.0]], dtype=np.float32),
            "processed_lens": np.array([8], dtype=np.int64),
        },
    ]
    fixtures = []
    names = ["x", "cache", "processed_lens"]
    for step, values in enumerate(step_values):
        step_name = f"step_{step:02d}"
        directory = fixtures_dir / "mindir" / step_name
        directory.mkdir(parents=True)
        entries = []
        for index, name in enumerate(names):
            array = values[name]
            path = directory / f"input_{index:02d}_{name}.bin"
            array.tofile(path)
            entries.append(array_entry(index, name, array, path, fixtures_dir))
        fixtures.append(
            {
                "name": step_name,
                "step": step,
                "processed_lens": [int(values["processed_lens"][0])],
                "inputs": entries,
            }
        )
    manifest = {
        "format_version": 1,
        "generator": "generate_zipformer_streaming_fixtures.py",
        "configuration": {
            "batch_size": 1,
            "feature_dim": 2,
            "input_frames": 4,
            "shift_frames": 2,
            "chunk_size": 1,
            "left_context_frames": 1,
            "state_advance_steps": 1,
            "fixture_count": 2,
            "seed": 42,
        },
        "features": [],
        "backends": {
            "mindir": {
                "backend": "mindir",
                "model": str(model_path.resolve()),
                "input_order": names,
                "fixtures": fixtures,
            }
        },
    }
    (fixtures_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return fixtures_dir, model_path, manifest


def write_manifest(fixtures_dir: Path, manifest: Dict[str, Any]) -> None:
    (fixtures_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def run_cli(
    fixtures_dir: Path,
    fake_mslite: FakeMindSporeLite,
    *extra_args: str,
    warmup: int = 0,
    loops: int = 1,
) -> Tuple[str, str]:
    benchmark = load_benchmark(fake_mslite)
    argv = [
        str(BENCH_PATH),
        "--fixtures-dir",
        str(fixtures_dir),
        "--warmup",
        str(warmup),
        "--loops",
        str(loops),
        "--no-cpu-bind",
        *extra_args,
    ]
    stdout = io.StringIO()
    stderr = io.StringIO()
    with mock.patch.object(sys, "argv", argv), mock.patch.dict(
        sys.modules,
        {"mindspore_lite": fake_mslite},
    ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        benchmark.main()
    return stdout.getvalue(), stderr.getvalue()


class MindIRFixtureBenchmarkTests(unittest.TestCase):
    maxDiff = None

    def test_cpu_precision_legacy_fallback_and_affinity_result(self) -> None:
        runtime = FakeMindSporeLite()
        benchmark = load_benchmark(runtime)
        legacy_cpu = types.SimpleNamespace(enable_fp16=None)

        self.assertEqual(
            benchmark.set_cpu_precision(legacy_cpu, True),
            "preferred_fp16",
        )
        self.assertTrue(legacy_cpu.enable_fp16)

        with mock.patch.object(benchmark.platform, "system", return_value="Darwin"):
            self.assertFalse(benchmark.bind_cpu(3))
        with mock.patch.object(benchmark.platform, "system", return_value="Linux"), mock.patch.object(
            benchmark.os,
            "sched_setaffinity",
            side_effect=OSError("not allowed"),
            create=True,
        ):
            self.assertFalse(benchmark.bind_cpu(3))

    def test_config_path_is_forwarded_to_model_build(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixtures_dir, _, _ = make_fixtures(root)
            config_path = root / "ascend.ini"
            config_path.write_text("[ascend_context]\n", encoding="utf-8")
            runtime = FakeMindSporeLite()

            run_cli(
                fixtures_dir,
                runtime,
                "--config-path",
                str(config_path),
                "--output-dir",
                str(root / "output"),
            )

            positional, keyword = runtime.models[0].build_calls[0]
            self.assertFalse(keyword)
            self.assertEqual(len(positional), 4)
            self.assertEqual(positional[3], str(config_path.resolve()))

    def test_cpu_run_resizes_refreshes_handles_and_never_feeds_outputs_back(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixtures_dir, _, _ = make_fixtures(root)
            output_dir = root / "cpu-output"
            runtime = FakeMindSporeLite()

            run_cli(
                fixtures_dir,
                runtime,
                "--device",
                "cpu",
                "--threads",
                "3",
                "--enable-fp16",
                "--output-dir",
                str(output_dir),
                warmup=1,
                loops=2,
            )

            self.assertEqual(len(runtime.contexts), 1)
            context = runtime.contexts[0]
            self.assertEqual(context.target, ["cpu"])
            self.assertEqual(context.cpu.thread_num, 3)
            self.assertTrue(
                context.cpu.enable_fp16 is True
                or context.cpu.precision_mode in {"preferred_fp16", "enforce_fp16"}
            )
            self.assertEqual(len(runtime.models), 1)
            model = runtime.models[0]
            self.assertEqual(
                model.resize_calls,
                [[[1, 4, 2], [1, 2], [1]]],
            )

            # Per fixture: one warmup + two timed calls + one un-timed output call.
            self.assertEqual(len(model.predict_snapshots), 8)
            first_step = model.predict_snapshots[:4]
            second_step = model.predict_snapshots[4:]
            for snapshot in first_step:
                np.testing.assert_array_equal(
                    snapshot["cache"],
                    np.array([[10.0, 11.0]], dtype=np.float32),
                )
                np.testing.assert_array_equal(snapshot["processed_lens"], [4])
            for snapshot in second_step:
                np.testing.assert_array_equal(
                    snapshot["cache"],
                    np.array([[20.0, 21.0]], dtype=np.float32),
                )
                np.testing.assert_array_equal(snapshot["processed_lens"], [8])

            results_path = output_dir / "outputs_manifest.json"
            self.assertTrue(results_path.is_file())
            results = json.loads(results_path.read_text(encoding="utf-8"))
            self.assertEqual(results["format_version"], 1)
            self.assertEqual(results["configuration"]["device"], "cpu")
            self.assertEqual(results["configuration"]["warmup"], 1)
            self.assertEqual(results["configuration"]["loops"], 2)
            self.assertEqual(results["configuration"]["timing_scope"], "model.predict")
            self.assertEqual(len(results["steps"]), 2)
            for step in results["steps"]:
                self.assertEqual(step["runs"], 2)
                self.assertEqual(
                    set(step["latency_ms"]),
                    {"mean", "std", "min", "p50", "p90", "p95", "p99", "max"},
                )
                self.assertEqual(len(step["outputs"]), 3)
                for output in step["outputs"]:
                    path = output_dir / output["file"]
                    self.assertTrue(path.is_file())
                    self.assertEqual(path.stat().st_size, output["byte_size"])

    def test_ascend_context_records_explicit_precision_and_provider(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixtures_dir, _, _ = make_fixtures(root)
            output_dir = root / "ascend-output"
            runtime = FakeMindSporeLite()

            run_cli(
                fixtures_dir,
                runtime,
                "--device",
                "ascend",
                "--device-id",
                "5",
                "--threads",
                "4",
                "--ascend-precision-mode",
                "enforce_fp16",
                "--ascend-provider",
                "ge",
                "--output-dir",
                str(output_dir),
            )

            context = runtime.contexts[0]
            self.assertEqual(context.target, ["ascend"])
            self.assertEqual(context.cpu.thread_num, 4)
            self.assertEqual(context.ascend.device_id, 5)
            self.assertEqual(context.ascend.precision_mode, "enforce_fp16")
            self.assertEqual(context.ascend.provider, "ge")
            results = json.loads(
                (output_dir / "outputs_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(results["configuration"]["device"], "ascend")
            self.assertEqual(results["configuration"]["device_id"], 5)
            self.assertEqual(
                results["configuration"]["requested_ascend_precision_mode"],
                "enforce_fp16",
            )
            self.assertEqual(
                results["configuration"]["context_ascend_precision_mode"],
                "enforce_fp16",
            )
            self.assertEqual(
                results["configuration"]["requested_ascend_provider"], "ge"
            )
            self.assertEqual(
                results["configuration"]["context_ascend_provider"], "ge"
            )

    def test_ascend_precision_is_left_unset_when_not_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixtures_dir, _, _ = make_fixtures(root)
            output_dir = root / "ascend-default-output"
            runtime = FakeMindSporeLite()

            _, stderr = run_cli(
                fixtures_dir,
                runtime,
                "--device",
                "ascend",
                "--output-dir",
                str(output_dir),
            )

            self.assertIsNone(runtime.contexts[0].ascend.precision_mode)
            self.assertIn("runtime default is left unchanged", stderr)
            results = json.loads(
                (output_dir / "outputs_manifest.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(
                results["configuration"]["requested_ascend_precision_mode"]
            )
            self.assertIsNone(
                results["configuration"]["context_ascend_precision_mode"]
            )
            self.assertEqual(
                results["configuration"][
                    "expected_runtime_default_ascend_precision_mode"
                ],
                "enforce_fp16",
            )

    def test_manifest_and_raw_inputs_are_validated_strictly(self) -> None:
        def wrong_format(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            manifest["format_version"] = 2

        def wrong_count(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            manifest["configuration"]["fixture_count"] = 3

        def duplicate_step(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            manifest["backends"]["mindir"]["fixtures"][1]["step"] = 0

        def unsafe_step_name(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            manifest["backends"]["mindir"]["fixtures"][0]["name"] = "../escape"

        def non_contiguous_index(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            manifest["backends"]["mindir"]["fixtures"][0]["inputs"][1]["index"] = 9

        def wrong_input_order(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            inputs = manifest["backends"]["mindir"]["fixtures"][0]["inputs"]
            inputs[0], inputs[1] = inputs[1], inputs[0]

        def invalid_dtype(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            manifest["backends"]["mindir"]["fixtures"][0]["inputs"][0]["dtype_str"] = "invalid"

        def wrong_element_count(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            manifest["backends"]["mindir"]["fixtures"][0]["inputs"][0]["element_count"] += 1

        def wrong_manifest_byte_size(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            manifest["backends"]["mindir"]["fixtures"][0]["inputs"][0]["byte_size"] += 1

        def wrong_actual_file_size(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            entry = manifest["backends"]["mindir"]["fixtures"][0]["inputs"][0]
            with (fixtures_dir / entry["file"]).open("ab") as stream:
                stream.write(b"\0")

        def inconsistent_shapes(manifest: Dict[str, Any], fixtures_dir: Path) -> None:
            entry = manifest["backends"]["mindir"]["fixtures"][1]["inputs"][0]
            entry["shape"] = [1, 2, 4]

        mutations = {
            "unsupported format": wrong_format,
            "fixture count mismatch": wrong_count,
            "duplicate steps": duplicate_step,
            "unsafe step name": unsafe_step_name,
            "non-contiguous input indexes": non_contiguous_index,
            "wrong input order": wrong_input_order,
            "invalid dtype": invalid_dtype,
            "shape/element count mismatch": wrong_element_count,
            "manifest byte size mismatch": wrong_manifest_byte_size,
            "actual file byte size mismatch": wrong_actual_file_size,
            "fixture shapes differ": inconsistent_shapes,
        }

        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary_directory:
                root = Path(temporary_directory)
                fixtures_dir, _, base_manifest = make_fixtures(root)
                manifest = copy.deepcopy(base_manifest)
                mutate(manifest, fixtures_dir)
                write_manifest(fixtures_dir, manifest)
                runtime = FakeMindSporeLite()
                with self.assertRaises(SystemExit) as caught:
                    run_cli(fixtures_dir, runtime)
                self.assertEqual(caught.exception.code, 1)

    def test_model_input_order_shape_and_dtype_must_match_fixtures(self) -> None:
        runtimes = {
            "order": FakeMindSporeLite(
                [
                    ("cache", [1, 2], "float32"),
                    ("x", [1, 4, 2], "float32"),
                    ("processed_lens", [1], "int64"),
                ]
            ),
            "shape": FakeMindSporeLite(
                [
                    ("x", [1, 3, 2], "float32"),
                    ("cache", [1, 2], "float32"),
                    ("processed_lens", [1], "int64"),
                ]
            ),
            "dtype": FakeMindSporeLite(
                [
                    ("x", [1, 4, 2], "float16"),
                    ("cache", [1, 2], "float32"),
                    ("processed_lens", [1], "int64"),
                ]
            ),
        }
        for label, runtime in runtimes.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary_directory:
                fixtures_dir, _, _ = make_fixtures(Path(temporary_directory))
                with self.assertRaises(SystemExit) as caught:
                    run_cli(fixtures_dir, runtime)
                self.assertEqual(caught.exception.code, 1)

    def test_nonempty_output_directory_requires_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixtures_dir, _, _ = make_fixtures(root)
            output_dir = root / "existing-output"
            output_dir.mkdir()
            marker = output_dir / "keep.txt"
            marker.write_text("user data", encoding="utf-8")
            runtime = FakeMindSporeLite()

            with self.assertRaises(SystemExit) as caught:
                run_cli(
                    fixtures_dir,
                    runtime,
                    "--output-dir",
                    str(output_dir),
                )
            self.assertEqual(caught.exception.code, 1)
            self.assertEqual(marker.read_text(encoding="utf-8"), "user data")
            self.assertFalse(runtime.models)


if __name__ == "__main__":
    unittest.main()
