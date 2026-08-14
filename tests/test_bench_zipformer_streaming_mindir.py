from __future__ import annotations

import contextlib
import importlib.util
import io
import shlex
import sys
import tempfile
import types
import unittest
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCH_PATH = ROOT / "scripts" / "bench_zipformer_streaming_mindir.py"


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
        self.values = (
            np.zeros(self.shape, dtype=np.dtype(dtype))
            if values is None
            else np.array(values, copy=True)
        )

    def set_data_from_numpy(self, values: np.ndarray) -> None:
        array = np.asarray(values)
        if list(array.shape) != self.shape:
            raise AssertionError(
                f"Unexpected shape for {self.name}: {list(array.shape)} != {self.shape}"
            )
        self.values = np.array(array, copy=True)

    def get_data_to_numpy(self) -> np.ndarray:
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
    def __init__(self) -> None:
        self.inputs = [
            FakeTensor("x", [1, 4, 2], "float32"),
            FakeTensor("cache", [1, 2], "float32"),
            FakeTensor("processed_lens", [1], "int64"),
        ]
        self.build_calls: List[Tuple[Any, ...]] = []
        self.predict_snapshots: List[Dict[str, np.ndarray]] = []

    def build_from_file(self, *args: Any) -> None:
        self.build_calls.append(args)

    def get_inputs(self) -> List[FakeTensor]:
        return list(self.inputs)

    def resize(self, inputs: Sequence[FakeTensor], shapes: Sequence[Sequence[int]]) -> None:
        raise AssertionError("The fixed-shape test model must not be resized.")

    def predict(self, inputs: Sequence[FakeTensor]) -> List[FakeTensor]:
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
            FakeTensor("new_cache", cache.shape, "float32", cache + 1.0),
            FakeTensor(
                "new_processed_lens",
                processed_lens.shape,
                "int64",
                processed_lens + 1,
            ),
        ]


class FakeMindSporeLite(types.ModuleType):
    def __init__(self) -> None:
        super().__init__("mindspore_lite")
        self.ModelType = types.SimpleNamespace(MINDIR="MINDIR")
        self.contexts: List[FakeContext] = []
        self.models: List[FakeModel] = []
        self.Context = self._new_context
        self.Model = self._new_model

    def _new_context(self) -> FakeContext:
        context = FakeContext()
        self.contexts.append(context)
        return context

    def _new_model(self) -> FakeModel:
        model = FakeModel()
        self.models.append(model)
        return model


def load_benchmark() -> Any:
    module_name = f"bench_zipformer_streaming_mindir_test_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(module_name, BENCH_PATH)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {BENCH_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def parse_args(module: Any, *arguments: str) -> Any:
    with mock.patch.object(sys, "argv", [str(BENCH_PATH), *arguments]):
        return module.parse_args()


class BenchStreamingMindIRTest(unittest.TestCase):
    def test_cpu_rejects_ascend_profile(self) -> None:
        module = load_benchmark()
        args = parse_args(module, "--model", "encoder.mindir", "--profile")
        with self.assertRaisesRegex(ValueError, "requires --device ascend"):
            module.validate_args(args)

    def test_cpu_context_remains_the_default(self) -> None:
        module = load_benchmark()
        fake_mslite = FakeMindSporeLite()
        args = parse_args(
            module, "--model", "encoder.mindir", "--enable-fp16", "--threads", "4"
        )
        module.validate_args(args)
        context, metadata = module.create_context(fake_mslite, args)

        self.assertEqual(context.target, ["cpu"])
        self.assertEqual(context.cpu.thread_num, 4)
        self.assertEqual(context.cpu.precision_mode, "preferred_fp16")
        self.assertIsNone(context.ascend.device_id)
        self.assertEqual(metadata["device"], "cpu")
        self.assertFalse(metadata["cpu_fallback_enabled"])

    def test_npu_alias_builds_and_runs_with_ascend_context(self) -> None:
        module = load_benchmark()
        fake_mslite = FakeMindSporeLite()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = root / "encoder.mindir"
            config_path = root / "config.ini"
            model_path.write_bytes(b"fake mindir")
            config_path.write_text("[ascend_context]\n", encoding="utf-8")
            argv = [
                str(BENCH_PATH),
                "--model",
                str(model_path),
                "--device",
                "npu",
                "--device-id",
                "3",
                "--ascend-precision-mode",
                "enforce_fp16",
                "--ascend-provider",
                "ge",
                "--config-path",
                str(config_path),
                "--feature-dim",
                "2",
                "--chunk-size",
                "1",
                "--left-context-frames",
                "1",
                "--warmup",
                "1",
                "--loops",
                "2",
                "--threads",
                "2",
                "--no-cpu-bind",
            ]
            stdout = io.StringIO()
            stderr = io.StringIO()
            with mock.patch.object(sys, "argv", argv), mock.patch.dict(
                sys.modules, {"mindspore_lite": fake_mslite}
            ):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(
                    stderr
                ):
                    module.main()

        context = fake_mslite.contexts[0]
        model = fake_mslite.models[0]
        self.assertEqual(context.target, ["ascend"])
        self.assertEqual(context.ascend.device_id, 3)
        self.assertEqual(context.ascend.precision_mode, "enforce_fp16")
        self.assertEqual(context.ascend.provider, "ge")
        self.assertEqual(context.cpu.thread_num, 2)
        self.assertEqual(context.cpu.precision_mode, "enforce_fp32")
        self.assertEqual(
            model.build_calls,
            [
                (
                    str(model_path.resolve()),
                    "MINDIR",
                    context,
                    str(config_path.resolve()),
                )
            ],
        )
        self.assertEqual(len(model.predict_snapshots), 4)
        np.testing.assert_array_equal(
            model.predict_snapshots[2]["cache"], np.zeros((1, 2), np.float32)
        )
        np.testing.assert_array_equal(
            model.predict_snapshots[3]["cache"], np.ones((1, 2), np.float32)
        )
        self.assertIn("device=ascend", stdout.getvalue())
        self.assertIn("does not prove full NPU offload", stderr.getvalue())

    def test_msprof_command_relaunches_with_absolute_paths(self) -> None:
        module = load_benchmark()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            model_path = (root / "encoder.mindir").resolve()
            config_path = (root / "config.ini").resolve()
            profile_output = root / "profile"
            args = parse_args(
                module,
                "--model",
                "relative.mindir",
                "--device",
                "npu",
                "--profile",
                "--profile-output",
                str(profile_output),
            )
            module.validate_args(args)
            with mock.patch.object(
                module.shutil, "which", return_value="/opt/ascend/bin/msprof"
            ):
                command, output_dir = module.create_msprof_command(
                    args,
                    model_path,
                    str(config_path),
                    argv=[
                        "--model",
                        "relative.mindir",
                        "--device",
                        "npu",
                        "--profile",
                    ],
                )

        self.assertEqual(command[0], "/opt/ascend/bin/msprof")
        self.assertEqual(command[2], f"--output={output_dir}")
        application = shlex.split(command[1].removeprefix("--application="))
        self.assertEqual(application[-5:], [
            "--model",
            str(model_path),
            "--profile-child",
            "--config-path",
            str(config_path),
        ])
        self.assertTrue(output_dir.is_absolute())


if __name__ == "__main__":
    unittest.main()
