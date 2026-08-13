#!/usr/bin/env python3
"""Benchmark streaming Zipformer MindIR fixtures on CPU or Ascend.

Each fixture produced by generate_zipformer_streaming_fixtures.py is a complete
input snapshot (features plus every cache/state tensor).  Fixtures are therefore
benchmarked independently; model outputs are never fed into a later fixture.

Examples:
    python scripts/bench_zipformer_streaming_mindir_fixtures.py \
        --fixtures-dir fixtures/zipformer --device cpu

    python scripts/bench_zipformer_streaming_mindir_fixtures.py \
        --fixtures-dir fixtures/zipformer --device ascend --device-id 0 \
        --ascend-precision-mode enforce_fp16
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


ASCEND_PRECISION_MODES = (
    "enforce_fp32",
    "preferred_fp32",
    "enforce_fp16",
    "enforce_origin",
    "preferred_optimal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixtures-dir",
        required=True,
        help="Fixture directory containing manifest.json.",
    )
    parser.add_argument(
        "--model",
        help="MindIR model path. Default: backends.mindir.model in the manifest.",
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "ascend", "npu"),
        default="cpu",
        help="Inference device; npu is an alias for ascend. Default: cpu.",
    )
    parser.add_argument(
        "--device-id",
        type=int,
        default=0,
        help="Ascend device ID. Default: 0.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="CPU thread count, including Ascend CPU fallback. Default: 1.",
    )
    parser.add_argument(
        "--enable-fp16",
        action="store_true",
        help="Prefer FP16 for CPU inference. Not valid for Ascend.",
    )
    parser.add_argument(
        "--ascend-precision-mode",
        choices=ASCEND_PRECISION_MODES,
        help=(
            "Ascend runtime precision mode. Keep it consistent with converter_lite; "
            "omitting this option leaves the runtime default unchanged."
        ),
    )
    parser.add_argument(
        "--ascend-provider",
        choices=("default", "ge"),
        default="default",
        help="Ascend provider. Default leaves the MindSpore Lite provider unchanged.",
    )
    parser.add_argument(
        "--config-path",
        help=(
            "Optional MindSpore Lite build configuration. Ascend dynamic inputs "
            "must declare supported shape gears in a backend-compatible config "
            "here or at conversion."
        ),
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Warmup predict calls per fixture. Default: 20.",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=100,
        help="Timed predict calls per fixture. Default: 100.",
    )
    parser.add_argument(
        "--cpu",
        type=int,
        default=0,
        help="Host CPU core to bind on Linux. Default: 0.",
    )
    parser.add_argument(
        "--no-cpu-bind",
        action="store_true",
        help="Do not bind the process to a host CPU core.",
    )
    parser.add_argument(
        "--output-dir",
        help="Output directory. Default: <fixtures-dir>/mindir_outputs/<device>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    return parser.parse_args()


def normalize_device(device: str) -> str:
    return "ascend" if device == "npu" else device


def validate_args(args: argparse.Namespace) -> None:
    args.device = normalize_device(args.device)
    if args.warmup < 0 or args.loops < 1:
        raise ValueError("warmup must be non-negative and loops must be positive.")
    if args.threads < 1 or args.cpu < 0 or args.device_id < 0:
        raise ValueError("threads must be positive; cpu and device-id must be non-negative.")
    if args.device == "cpu":
        if args.ascend_precision_mode is not None:
            raise ValueError("--ascend-precision-mode requires --device ascend.")
        if args.ascend_provider != "default":
            raise ValueError("--ascend-provider requires --device ascend.")
    elif args.enable_fp16:
        raise ValueError(
            "--enable-fp16 is CPU-only; use --ascend-precision-mode for Ascend."
        )
    if args.config_path and not Path(args.config_path).expanduser().is_file():
        raise FileNotFoundError(
            f"MindSpore Lite config file does not exist: {args.config_path}"
        )


def bind_cpu(cpu: int) -> bool:
    if platform.system() != "Linux":
        print(
            f"[warning] CPU affinity is unavailable on {platform.system()}; "
            "use the platform affinity tool if needed.",
            file=sys.stderr,
        )
        return False
    try:
        os.sched_setaffinity(0, {cpu})
        print(f"[info] Bound process to CPU {cpu}.")
        return True
    except OSError as error:
        print(f"[warning] Could not bind process to CPU {cpu}: {error}", file=sys.stderr)
        return False


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "tensor"


def require_dict(value: Any, description: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{description} must be an object.")
    return value


def require_nonempty_string(value: Any, description: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{description} must be a non-empty string.")
    return value


def require_plain_int(value: Any, description: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{description} must be an integer >= {minimum}.")
    return value


def fixture_file_path(fixtures_dir: Path, value: Any, description: str) -> Path:
    relative = Path(require_nonempty_string(value, description))
    if relative.is_absolute():
        raise ValueError(f"{description} must be relative to the fixture directory.")
    root = fixtures_dir.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{description} escapes the fixture directory: {relative}") from error
    return path


def validate_input_entry(
    fixtures_dir: Path,
    entry: Any,
    expected_index: int,
    expected_name: str,
    step_name: str,
) -> Tuple[Tuple[int, ...], np.dtype]:
    data = require_dict(entry, f"{step_name} input {expected_index}")
    index = require_plain_int(data.get("index"), f"{step_name} input index")
    if index != expected_index:
        raise ValueError(
            f"{step_name}: input indexes must be contiguous; expected {expected_index}, got {index}."
        )
    name = require_nonempty_string(data.get("name"), f"{step_name} input {index} name")
    if name != expected_name:
        raise ValueError(
            f"{step_name}: input {index} is {name!r}; manifest input_order expects {expected_name!r}."
        )
    raw_shape = data.get("shape")
    if not isinstance(raw_shape, list):
        raise ValueError(f"{step_name} input {name}: shape must be a list.")
    shape = tuple(
        require_plain_int(value, f"{step_name} input {name} shape[{axis}]", minimum=1)
        for axis, value in enumerate(raw_shape)
    )
    dtype_str = require_nonempty_string(
        data.get("dtype_str"), f"{step_name} input {name} dtype_str"
    )
    try:
        dtype = np.dtype(dtype_str)
    except TypeError as error:
        raise ValueError(
            f"{step_name} input {name}: invalid dtype_str {dtype_str!r}."
        ) from error
    declared_dtype = require_nonempty_string(
        data.get("dtype"), f"{step_name} input {name} dtype"
    )
    if declared_dtype != str(dtype):
        raise ValueError(
            f"{step_name} input {name}: dtype={declared_dtype!r} does not match "
            f"dtype_str={dtype_str!r} ({dtype})."
        )
    expected_elements = math.prod(shape)
    element_count = require_plain_int(
        data.get("element_count"), f"{step_name} input {name} element_count"
    )
    if element_count != expected_elements:
        raise ValueError(
            f"{step_name} input {name}: shape {list(shape)} contains {expected_elements} "
            f"elements, manifest declares {element_count}."
        )
    expected_bytes = expected_elements * dtype.itemsize
    byte_size = require_plain_int(
        data.get("byte_size"), f"{step_name} input {name} byte_size"
    )
    if byte_size != expected_bytes:
        raise ValueError(
            f"{step_name} input {name}: dtype/shape require {expected_bytes} bytes, "
            f"manifest declares {byte_size}."
        )
    path = fixture_file_path(
        fixtures_dir, data.get("file"), f"{step_name} input {name} file"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Fixture input file does not exist: {path}")
    actual_bytes = path.stat().st_size
    if actual_bytes != byte_size:
        raise ValueError(
            f"{step_name} input {name}: file has {actual_bytes} bytes; "
            f"manifest expects {byte_size}."
        )
    return shape, dtype


def load_manifest(fixtures_dir: Path) -> Dict[str, Any]:
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    manifest = require_dict(
        json.loads(manifest_path.read_text(encoding="utf-8")), "manifest"
    )
    if manifest.get("format_version") != 1:
        raise ValueError(
            f"Unsupported fixture format_version {manifest.get('format_version')!r}; expected 1."
        )
    configuration = require_dict(manifest.get("configuration"), "manifest.configuration")
    fixture_count = require_plain_int(
        configuration.get("fixture_count"),
        "manifest.configuration.fixture_count",
        minimum=1,
    )
    backends = require_dict(manifest.get("backends"), "manifest.backends")
    if "mindir" not in backends:
        raise ValueError(f"Manifest has no MindIR backend fixtures: {manifest_path}")
    backend = require_dict(backends["mindir"], "manifest.backends.mindir")
    if backend.get("backend") != "mindir":
        raise ValueError("manifest.backends.mindir.backend must be 'mindir'.")
    input_order = backend.get("input_order")
    if not isinstance(input_order, list) or not input_order:
        raise ValueError("manifest.backends.mindir.input_order must be a non-empty list.")
    input_order = [
        require_nonempty_string(name, f"MindIR input_order[{index}]")
        for index, name in enumerate(input_order)
    ]
    if len(set(input_order)) != len(input_order):
        raise ValueError("MindIR input_order contains duplicate names.")
    fixtures = backend.get("fixtures")
    if not isinstance(fixtures, list) or len(fixtures) != fixture_count:
        actual = len(fixtures) if isinstance(fixtures, list) else "not a list"
        raise ValueError(
            f"MindIR fixture count is {actual}; configuration.fixture_count is {fixture_count}."
        )

    names: List[str] = []
    steps: List[int] = []
    expected_signature: Optional[Tuple[Tuple[str, Tuple[int, ...], str], ...]] = None
    for fixture_index, raw_fixture in enumerate(fixtures):
        fixture = require_dict(raw_fixture, f"MindIR fixture {fixture_index}")
        name = require_nonempty_string(fixture.get("name"), f"MindIR fixture {fixture_index} name")
        if safe_name(name) != name:
            raise ValueError(
                f"MindIR fixture {fixture_index} name {name!r} is not a safe path component."
            )
        step = require_plain_int(fixture.get("step"), f"MindIR fixture {name} step")
        inputs = fixture.get("inputs")
        if not isinstance(inputs, list) or len(inputs) != len(input_order):
            actual = len(inputs) if isinstance(inputs, list) else "not a list"
            raise ValueError(
                f"{name}: has {actual} inputs; input_order contains {len(input_order)}."
            )
        signature_entries = []
        for input_index, (entry, expected_name) in enumerate(zip(inputs, input_order)):
            shape, dtype = validate_input_entry(
                fixtures_dir, entry, input_index, expected_name, name
            )
            signature_entries.append((expected_name, shape, dtype.str))
        signature = tuple(signature_entries)
        if expected_signature is None:
            expected_signature = signature
        elif signature != expected_signature:
            raise ValueError(
                f"{name}: input names, shapes, or dtypes differ from the first fixture. "
                "Per-step resize is intentionally unsupported because it contaminates latency."
            )
        names.append(name)
        steps.append(step)

    if len(set(names)) != len(names):
        raise ValueError("MindIR fixture names must be unique.")
    if steps != list(range(fixture_count)):
        raise ValueError(
            f"MindIR fixture steps must be contiguous from 0; found {steps}."
        )
    return manifest


def resolve_model_path(args: argparse.Namespace, backend: Dict[str, Any]) -> Path:
    value = args.model if args.model else backend.get("model")
    model_path = Path(require_nonempty_string(value, "MindIR model path")).expanduser().resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    return model_path


def set_cpu_precision(cpu_context: Any, enable_fp16: bool) -> str:
    precision_mode = "preferred_fp16" if enable_fp16 else "enforce_fp32"
    if hasattr(cpu_context, "precision_mode"):
        try:
            cpu_context.precision_mode = precision_mode
            return precision_mode
        except (AttributeError, RuntimeError, ValueError):
            pass
    if hasattr(cpu_context, "enable_fp16"):
        cpu_context.enable_fp16 = enable_fp16
        return "preferred_fp16" if enable_fp16 else "enforce_fp32"
    raise RuntimeError("This MindSpore Lite Context exposes no CPU precision setting.")


def create_context(mslite: Any, args: argparse.Namespace) -> Tuple[Any, Dict[str, Any]]:
    context = mslite.Context()
    context.cpu.thread_num = args.threads
    cpu_precision_mode = set_cpu_precision(
        context.cpu, args.enable_fp16 if args.device == "cpu" else False
    )
    context.target = [args.device]

    context_precision = None
    context_provider = None
    if args.device == "ascend":
        context.ascend.device_id = args.device_id
        if args.ascend_precision_mode is not None:
            context.ascend.precision_mode = args.ascend_precision_mode
        else:
            print(
                "[warning] --ascend-precision-mode was not specified; the MindSpore "
                "Lite runtime default is left unchanged. For reproducible results, pass "
                "the mode used by converter_lite (its default is enforce_fp16).",
                file=sys.stderr,
            )
        if args.ascend_provider == "ge":
            context.ascend.provider = "ge"
        context_precision = getattr(context.ascend, "precision_mode", None)
        context_provider = getattr(context.ascend, "provider", None)

    metadata = {
        "device": args.device,
        "target": list(getattr(context, "target", [args.device])),
        "device_id": args.device_id if args.device == "ascend" else None,
        "threads": args.threads,
        "enable_fp16": args.enable_fp16,
        "cpu_precision_mode": cpu_precision_mode,
        "requested_ascend_precision_mode": args.ascend_precision_mode,
        "context_ascend_precision_mode": context_precision,
        "expected_runtime_default_ascend_precision_mode": (
            "enforce_fp16"
            if args.device == "ascend" and args.ascend_precision_mode is None
            else None
        ),
        "requested_ascend_provider": args.ascend_provider,
        "context_ascend_provider": context_provider,
        "cpu_fallback_enabled": args.device == "ascend",
    }
    return context, metadata


def numpy_dtype(tensor: Any) -> np.dtype:
    data_type = str(getattr(tensor, "dtype", "")).lower()
    if "bfloat16" in data_type:
        raise ValueError(
            f"Unsupported MindSpore Lite tensor type for NumPy fixture loading: {tensor.dtype}"
        )
    mappings = (
        ("float16", np.float16),
        ("float32", np.float32),
        ("float64", np.float64),
        ("double", np.float64),
        ("uint64", np.uint64),
        ("uint32", np.uint32),
        ("uint16", np.uint16),
        ("uint8", np.uint8),
        ("int64", np.int64),
        ("int32", np.int32),
        ("int16", np.int16),
        ("int8", np.int8),
        ("bool", np.bool_),
    )
    for token, dtype in mappings:
        if token in data_type:
            return np.dtype(dtype)
    if data_type.endswith("float") or data_type == "float":
        return np.dtype(np.float32)
    raise ValueError(f"Unsupported MindSpore Lite tensor type: {getattr(tensor, 'dtype', None)}")


def fixture_shapes_and_dtypes(
    backend: Dict[str, Any],
) -> Tuple[List[List[int]], List[np.dtype]]:
    entries = backend["fixtures"][0]["inputs"]
    return (
        [list(entry["shape"]) for entry in entries],
        [np.dtype(entry["dtype_str"]) for entry in entries],
    )


def check_model_input_names(inputs: Sequence[Any], input_order: Sequence[str]) -> None:
    model_names = [tensor.name for tensor in inputs]
    if model_names != list(input_order):
        raise ValueError(
            f"Model input order {model_names} does not match fixture input_order {list(input_order)}."
        )


def prepare_model_inputs(model: Any, backend: Dict[str, Any]) -> List[Any]:
    input_order = backend["input_order"]
    target_shapes, fixture_dtypes = fixture_shapes_and_dtypes(backend)
    inputs = list(model.get_inputs())
    check_model_input_names(inputs, input_order)
    if len(inputs) != len(target_shapes):
        raise ValueError(
            f"Model has {len(inputs)} inputs; fixtures contain {len(target_shapes)}."
        )

    needs_resize = False
    for tensor, target_shape in zip(inputs, target_shapes):
        model_shape = list(tensor.shape)
        if len(model_shape) != len(target_shape):
            raise ValueError(
                f"Model input {tensor.name} rank {len(model_shape)} does not match "
                f"fixture rank {len(target_shape)}."
            )
        for axis, (model_dim, target_dim) in enumerate(zip(model_shape, target_shape)):
            try:
                fixed_dim = int(model_dim)
            except (TypeError, ValueError):
                fixed_dim = 0
            if fixed_dim > 0 and fixed_dim != target_dim:
                raise ValueError(
                    f"Model input {tensor.name} shape {model_shape} conflicts with fixture "
                    f"shape {target_shape} at axis {axis}."
                )
            needs_resize = needs_resize or fixed_dim <= 0

    if needs_resize:
        try:
            model.resize(inputs, target_shapes)
        except RuntimeError as error:
            raise RuntimeError(
                f"Could not resize model inputs to fixture shapes {target_shapes}."
            ) from error
        inputs = list(model.get_inputs())
        check_model_input_names(inputs, input_order)

    for tensor, target_shape, fixture_dtype in zip(
        inputs, target_shapes, fixture_dtypes
    ):
        if list(tensor.shape) != target_shape:
            raise ValueError(
                f"Model input {tensor.name} shape is {list(tensor.shape)} after build/resize; "
                f"fixture shape is {target_shape}."
            )
        model_dtype = numpy_dtype(tensor)
        if model_dtype != fixture_dtype:
            raise ValueError(
                f"Model input {tensor.name} dtype is {model_dtype}; fixture dtype is "
                f"{fixture_dtype}. Refusing an implicit conversion."
            )
    return inputs


def load_fixture_inputs(fixtures_dir: Path, fixture: Dict[str, Any]) -> List[np.ndarray]:
    arrays = []
    for entry in fixture["inputs"]:
        path = fixture_file_path(
            fixtures_dir, entry["file"], f"{fixture['name']} input {entry['name']} file"
        )
        dtype = np.dtype(entry["dtype_str"])
        array = np.fromfile(path, dtype=dtype)
        if array.size != entry["element_count"] or array.nbytes != entry["byte_size"]:
            raise ValueError(
                f"Fixture input changed after validation: {path}; regenerate or copy it atomically."
            )
        arrays.append(np.ascontiguousarray(array.reshape(entry["shape"])))
    return arrays


def set_model_inputs(
    tensors: Sequence[Any], arrays: Sequence[np.ndarray], step_name: str
) -> None:
    if len(tensors) != len(arrays):
        raise ValueError(
            f"{step_name}: loaded {len(arrays)} arrays for {len(tensors)} model inputs."
        )
    for tensor, array in zip(tensors, arrays):
        expected_dtype = numpy_dtype(tensor)
        if list(tensor.shape) != list(array.shape) or expected_dtype != array.dtype:
            raise ValueError(
                f"{step_name}: input {tensor.name} is {array.dtype} {list(array.shape)}, "
                f"model expects {expected_dtype} {list(tensor.shape)}."
            )
        tensor.set_data_from_numpy(array)


def summarize(values: List[float]) -> Dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "p50": float(np.percentile(values, 50)),
        "p90": float(np.percentile(values, 90)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "max": max(values),
    }


def bench_step(
    model: Any, inputs: List[Any], warmup: int, loops: int
) -> Tuple[List[float], List[Any]]:
    for _ in range(warmup):
        model.predict(inputs)
    latencies_ms = []
    for _ in range(loops):
        started = time.perf_counter()
        model.predict(inputs)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    outputs = list(model.predict(inputs))
    return latencies_ms, outputs


def save_outputs(
    output_dir: Path, step_name: str, outputs: Sequence[Any]
) -> Dict[str, Any]:
    directory = output_dir / step_name
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, tensor in enumerate(outputs):
        name = str(getattr(tensor, "name", "")) or f"output_{index}"
        array = np.ascontiguousarray(
            np.array(tensor.get_data_to_numpy(), copy=True)
        )
        path = directory / f"output_{index:02d}_{safe_name(name)}.bin"
        array.tofile(path)
        entries.append(
            {
                "index": index,
                "name": name,
                "dtype": str(array.dtype),
                "dtype_str": array.dtype.str,
                "shape": list(array.shape),
                "element_count": int(array.size),
                "byte_size": int(array.nbytes),
                "file": str(path.relative_to(output_dir)),
            }
        )
    return {"name": step_name, "outputs": entries}


def print_step_summary(step_name: str, latencies_ms: List[float]) -> Dict[str, float]:
    summary = summarize(latencies_ms)
    print(f"\n[result] {step_name}: latency over {len(latencies_ms)} runs (ms)")
    for name in ("mean", "std", "min", "p50", "p90", "p95", "p99", "max"):
        print(f"  {name:>4}: {summary[name]:.3f}")
    return summary


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ValueError(
            f"Output directory is not empty: {path}. Use --overwrite to write into it."
        )
    path.mkdir(parents=True, exist_ok=True)


def mindspore_lite_version(mslite: Any) -> str:
    return str(getattr(mslite, "__version__", "unknown"))


def build_model(
    mslite: Any, model_path: Path, context: Any, config_path: Optional[str]
) -> Any:
    model = mslite.Model()
    if config_path:
        model.build_from_file(
            str(model_path),
            mslite.ModelType.MINDIR,
            context,
            str(Path(config_path).expanduser().resolve()),
        )
    else:
        model.build_from_file(str(model_path), mslite.ModelType.MINDIR, context)
    return model


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        fixtures_dir = Path(args.fixtures_dir).expanduser().resolve()
        manifest = load_manifest(fixtures_dir)
        backend = manifest["backends"]["mindir"]
        model_path = resolve_model_path(args, backend)
        output_dir = (
            Path(args.output_dir).expanduser().resolve()
            if args.output_dir
            else fixtures_dir / "mindir_outputs" / args.device
        )
        prepare_output_dir(output_dir, args.overwrite)
        cpu_bind_requested = not args.no_cpu_bind
        cpu_bind_succeeded = bind_cpu(args.cpu) if cpu_bind_requested else False
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))

        try:
            import mindspore_lite as mslite
        except ImportError as error:
            raise ImportError(
                "MindIR fixture benchmarking requires the platform-matched "
                "mindspore_lite package."
            ) from error

        context, context_metadata = create_context(mslite, args)
        model = build_model(mslite, model_path, context, args.config_path)
        inputs = prepare_model_inputs(model, backend)
        print(f"[info] Model: {model_path}")
        print(f"[info] Fixtures: {fixtures_dir} ({len(backend['fixtures'])} steps)")
        print(
            f"[info] device={args.device}, warmup={args.warmup}, loops={args.loops}, "
            f"threads={args.threads}"
        )
        if args.device == "ascend":
            print(
                "[warning] An Ascend target may retain CPU fallback for some nodes; "
                "successful execution alone does not prove full NPU offload.",
                file=sys.stderr,
            )

        steps = []
        for fixture in backend["fixtures"]:
            step_name = fixture["name"]
            arrays = load_fixture_inputs(fixtures_dir, fixture)
            set_model_inputs(inputs, arrays, step_name)
            latencies_ms, outputs = bench_step(
                model, inputs, args.warmup, args.loops
            )
            summary = print_step_summary(step_name, latencies_ms)
            record = save_outputs(output_dir, step_name, outputs)
            record.update(
                {
                    "step": fixture["step"],
                    "processed_lens": fixture.get("processed_lens"),
                    "latency_ms": summary,
                    "runs": len(latencies_ms),
                }
            )
            steps.append(record)

        results = {
            "format_version": 1,
            "generator": Path(__file__).name,
            "model": str(model_path),
            "fixtures_dir": str(fixtures_dir),
            "configuration": {
                "warmup": args.warmup,
                "loops": args.loops,
                "timing_scope": "model.predict",
                "mindspore_lite_version": mindspore_lite_version(mslite),
                "config_path": (
                    str(Path(args.config_path).expanduser().resolve())
                    if args.config_path
                    else None
                ),
                "cpu_bind_requested": cpu_bind_requested,
                "cpu_bind_enabled": cpu_bind_succeeded,
                "requested_cpu": args.cpu if cpu_bind_requested else None,
                "cpu": args.cpu if cpu_bind_succeeded else None,
                **context_metadata,
            },
            "steps": steps,
        }
        results_path = output_dir / "outputs_manifest.json"
        results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n[info] Wrote outputs and latency summary: {results_path}")
    except (FileNotFoundError, ImportError, OSError, RuntimeError, ValueError) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
