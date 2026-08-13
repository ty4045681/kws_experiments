from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List, Optional, Sequence
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
GENERATOR_PATH = ROOT / "scripts" / "generate_zipformer_streaming_fixtures.py"


def load_generator() -> Any:
    spec = importlib.util.spec_from_file_location(
        "generate_zipformer_streaming_fixtures_test",
        GENERATOR_PATH,
    )
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not import {GENERATOR_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load_generator()


def feature_args(
    *,
    batch_size: int,
    input_frames: int = 10,
    chunk_size: int = 2,
    feature_dim: int = 4,
    seed: int = 42,
) -> argparse.Namespace:
    return argparse.Namespace(
        batch_size=batch_size,
        input_frames=input_frames,
        chunk_size=chunk_size,
        feature_dim=feature_dim,
        seed=seed,
    )


class FakeOnnxMeta:
    def __init__(
        self,
        name: str,
        shape: Sequence[int],
        tensor_type: str = "tensor(float)",
    ) -> None:
        self.name = name
        self.shape = list(shape)
        self.type = tensor_type


class FakeOnnxSession:
    def __init__(
        self,
        inputs: Sequence[FakeOnnxMeta],
        outputs: Sequence[FakeOnnxMeta],
    ) -> None:
        self._inputs = list(inputs)
        self._outputs = list(outputs)

    def get_inputs(self) -> List[FakeOnnxMeta]:
        return list(self._inputs)

    def get_outputs(self) -> List[FakeOnnxMeta]:
        return list(self._outputs)


class FakeMindIRTensor:
    def __init__(self, name: str, shape: Sequence[int], dtype: str = "float32") -> None:
        self.name = name
        self.shape = list(shape)
        self.dtype = dtype


class FakeMindIRContext:
    def __init__(self) -> None:
        self.target: List[str] = []
        self.cpu = SimpleNamespace(thread_num=None, precision_mode=None)
        self.ascend = SimpleNamespace(device_id=None, precision_mode=None)


class FakeMindSporeLite:
    __version__ = "2.10.0"
    Context = FakeMindIRContext


def onnx_inputs() -> List[FakeOnnxMeta]:
    return [
        FakeOnnxMeta("x", [1, 10, 4]),
        FakeOnnxMeta("x_lens", [1], "tensor(int64)"),
        FakeOnnxMeta("cache_a", [1, 2]),
        FakeOnnxMeta("cache_b", [1, 3]),
    ]


def mindir_inputs() -> List[FakeMindIRTensor]:
    return [
        FakeMindIRTensor("x", [1, 10, 4]),
        FakeMindIRTensor("x_lens", [1], "int64"),
        FakeMindIRTensor("cache_a", [1, 2]),
        FakeMindIRTensor("cache_b", [1, 3]),
    ]


def mindir_context_args(
    device: str,
    *,
    precision_mode: Optional[str] = None,
    device_id: int = 0,
    threads: int = 2,
) -> argparse.Namespace:
    return argparse.Namespace(
        mindir_device=device,
        device_id=device_id,
        ascend_precision_mode=precision_mode,
        threads=threads,
    )


class MindIRDeviceCliTests(unittest.TestCase):
    def test_parse_ascend_generation_options(self) -> None:
        argv = [
            str(GENERATOR_PATH),
            "--mindir-model",
            "encoder-ascend-oriented.mindir",
            "--output-dir",
            "fixtures/ascend",
            "--mindir-device",
            "ascend",
            "--device-id",
            "3",
            "--ascend-precision-mode",
            "enforce_fp16",
        ]

        with mock.patch.object(sys, "argv", argv):
            args = generator.parse_args()

        self.assertEqual(args.mindir_device, "ascend")
        self.assertEqual(args.device_id, 3)
        self.assertEqual(args.ascend_precision_mode, "enforce_fp16")

    def test_cpu_context_remains_the_default(self) -> None:
        context, metadata = generator.create_mindir_context(
            FakeMindSporeLite,
            mindir_context_args("cpu", threads=4),
        )

        self.assertEqual(context.target, ["cpu"])
        self.assertEqual(context.cpu.thread_num, 4)
        self.assertEqual(context.cpu.precision_mode, "enforce_fp32")
        self.assertEqual(metadata["device"], "cpu")
        self.assertIsNone(metadata["device_id"])
        self.assertFalse(metadata["cpu_fallback_enabled"])

    def test_ascend_context_sets_device_and_explicit_precision(self) -> None:
        context, metadata = generator.create_mindir_context(
            FakeMindSporeLite,
            mindir_context_args(
                "ascend",
                precision_mode="enforce_fp16",
                device_id=2,
            ),
        )

        self.assertEqual(context.target, ["ascend"])
        self.assertEqual(context.ascend.device_id, 2)
        self.assertEqual(context.ascend.precision_mode, "enforce_fp16")
        self.assertEqual(metadata["mindspore_lite_version"], "2.10.0")
        self.assertEqual(metadata["requested_ascend_precision_mode"], "enforce_fp16")
        self.assertEqual(metadata["context_ascend_precision_mode"], "enforce_fp16")
        self.assertIsNone(metadata["expected_runtime_default_ascend_precision_mode"])
        self.assertTrue(metadata["cpu_fallback_enabled"])

    def test_npu_alias_and_unspecified_precision_preserve_runtime_default(self) -> None:
        context, metadata = generator.create_mindir_context(
            FakeMindSporeLite,
            mindir_context_args("npu", precision_mode=None),
        )

        self.assertEqual(context.target, ["ascend"])
        self.assertIsNone(context.ascend.precision_mode)
        self.assertEqual(metadata["device"], "ascend")
        self.assertEqual(
            metadata["expected_runtime_default_ascend_precision_mode"],
            "enforce_fp16",
        )


class PreparedFeaturesTests(unittest.TestCase):
    def test_batch_one_preserves_legacy_random_values(self) -> None:
        args = feature_args(
            batch_size=1,
            input_frames=10,
            chunk_size=2,
            feature_dim=4,
            seed=1234,
        )
        fixture_count = 4
        shift_frames = 2 * args.chunk_size
        total_frames = args.input_frames + (fixture_count - 1) * shift_frames

        # This is the exact algorithm used before batch support was added.
        legacy_stream = np.random.default_rng(args.seed).standard_normal(
            (total_frames, args.feature_dim),
            dtype=np.float32,
        )
        expected = [
            np.ascontiguousarray(
                legacy_stream[
                    step * shift_frames:step * shift_frames + args.input_frames
                ][None]
            )
            for step in range(fixture_count)
        ]

        actual = generator.prepared_features(args, fixture_count)

        self.assertEqual(len(actual), fixture_count)
        for actual_window, expected_window in zip(actual, expected):
            self.assertEqual(actual_window.shape, (1, args.input_frames, args.feature_dim))
            self.assertTrue(actual_window.flags.c_contiguous)
            np.testing.assert_array_equal(actual_window, expected_window)

    def test_batch_windows_have_expected_shape_overlap_and_contiguity(self) -> None:
        args = feature_args(
            batch_size=3,
            input_frames=10,
            chunk_size=2,
            feature_dim=5,
            seed=5678,
        )
        fixture_count = 4
        shift_frames = 2 * args.chunk_size
        total_frames = args.input_frames + (fixture_count - 1) * shift_frames
        expected_stream = np.random.default_rng(args.seed).standard_normal(
            (args.batch_size, total_frames, args.feature_dim),
            dtype=np.float32,
        )

        windows = generator.prepared_features(args, fixture_count)

        self.assertEqual(len(windows), fixture_count)
        for step, window in enumerate(windows):
            self.assertEqual(
                window.shape,
                (args.batch_size, args.input_frames, args.feature_dim),
            )
            self.assertEqual(window.dtype, np.dtype(np.float32))
            self.assertTrue(window.flags.c_contiguous)
            np.testing.assert_array_equal(
                window,
                expected_stream[
                    :,
                    step * shift_frames:step * shift_frames + args.input_frames,
                    :,
                ],
            )

        overlap_frames = args.input_frames - shift_frames
        self.assertGreater(overlap_frames, 0)
        for previous, current in zip(windows, windows[1:]):
            np.testing.assert_array_equal(
                previous[:, shift_frames:, :],
                current[:, :overlap_frames, :],
            )


class OnnxStateMappingTests(unittest.TestCase):
    def test_complete_named_mapping_succeeds_and_preserves_output_indexes(self) -> None:
        outputs = [
            FakeOnnxMeta("encoder_out", [1, 1, 4]),
            FakeOnnxMeta("new_cache_b", [1, 3]),
            FakeOnnxMeta("new_cache_a", [1, 2]),
        ]
        session = FakeOnnxSession(onnx_inputs(), outputs)

        self.assertEqual(
            generator.onnx_state_mapping(session),
            [("cache_b", 1), ("cache_a", 2)],
        )

    def test_partial_named_mapping_is_rejected(self) -> None:
        outputs = [
            FakeOnnxMeta("encoder_out", [1, 1, 4]),
            FakeOnnxMeta("new_cache_a", [1, 2]),
            FakeOnnxMeta("cache_b_out", [1, 3]),
        ]
        session = FakeOnnxSession(onnx_inputs(), outputs)

        with self.assertRaisesRegex(
            ValueError,
            r"Incomplete ONNX new_\* state mapping: .*missing state outputs for \['cache_b'\]",
        ):
            generator.onnx_state_mapping(session)

    def test_duplicate_named_mapping_is_rejected(self) -> None:
        outputs = [
            FakeOnnxMeta("encoder_out", [1, 1, 4]),
            FakeOnnxMeta("new_cache_a", [1, 2]),
            FakeOnnxMeta("new_cache_a", [1, 2]),
            FakeOnnxMeta("new_cache_b", [1, 3]),
        ]
        session = FakeOnnxSession(onnx_inputs(), outputs)

        with self.assertRaisesRegex(
            ValueError,
            r"Incomplete ONNX new_\* state mapping: .*duplicate state outputs for \['cache_a'\]",
        ):
            generator.onnx_state_mapping(session)

    def test_completely_unnamed_states_use_validated_ordered_fallback(self) -> None:
        outputs = [
            FakeOnnxMeta("encoder_out", [1, 1, 4]),
            FakeOnnxMeta("state_out_0", [1, 2]),
            FakeOnnxMeta("state_out_1", [1, 3]),
        ]
        session = FakeOnnxSession(onnx_inputs(), outputs)
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            mapping = generator.onnx_state_mapping(session)

        self.assertEqual(mapping, [("cache_a", 1), ("cache_b", 2)])
        self.assertIn("using ordered state mapping", stderr.getvalue())


class MindIRStateMappingTests(unittest.TestCase):
    def test_complete_named_mapping_succeeds_and_preserves_tensor_indexes(self) -> None:
        outputs = [
            FakeMindIRTensor("encoder_out", [1, 1, 4]),
            FakeMindIRTensor("new_cache_b", [1, 3]),
            FakeMindIRTensor("new_cache_a", [1, 2]),
        ]

        self.assertEqual(
            generator.mindir_state_mapping(mindir_inputs(), outputs),
            [(3, 1), (2, 2)],
        )

    def test_partial_named_mapping_is_rejected(self) -> None:
        outputs = [
            FakeMindIRTensor("encoder_out", [1, 1, 4]),
            FakeMindIRTensor("new_cache_a", [1, 2]),
            FakeMindIRTensor("cache_b_out", [1, 3]),
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"Incomplete MindIR new_\* state mapping: .*missing state outputs for \['cache_b'\]",
        ):
            generator.mindir_state_mapping(mindir_inputs(), outputs)

    def test_duplicate_named_mapping_is_rejected(self) -> None:
        outputs = [
            FakeMindIRTensor("encoder_out", [1, 1, 4]),
            FakeMindIRTensor("new_cache_a", [1, 2]),
            FakeMindIRTensor("new_cache_a", [1, 2]),
            FakeMindIRTensor("new_cache_b", [1, 3]),
        ]

        with self.assertRaisesRegex(
            ValueError,
            r"Incomplete MindIR new_\* state mapping: .*duplicate state outputs for \['cache_a'\]",
        ):
            generator.mindir_state_mapping(mindir_inputs(), outputs)

    def test_completely_unnamed_states_use_validated_ordered_fallback(self) -> None:
        outputs = [
            FakeMindIRTensor("encoder_out", [1, 1, 4]),
            FakeMindIRTensor("state_out_0", [1, 2]),
            FakeMindIRTensor("state_out_1", [1, 3]),
        ]
        stderr = io.StringIO()

        with contextlib.redirect_stderr(stderr):
            mapping = generator.mindir_state_mapping(mindir_inputs(), outputs)

        self.assertEqual(mapping, [(2, 1), (3, 2)])
        self.assertIn("using ordered state mapping", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
