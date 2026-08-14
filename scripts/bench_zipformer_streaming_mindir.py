#!/usr/bin/env python3
"""Benchmark a stateful streaming Zipformer MindIR encoder on CPU or Ascend.

Examples:
    python scripts/bench_zipformer_streaming_mindir.py \
        --model path/to/encoder.mindir

    python scripts/bench_zipformer_streaming_mindir.py \
        --model path/to/encoder.mindir --device ascend --device-id 0 \
        --ascend-precision-mode enforce_fp16

    python scripts/bench_zipformer_streaming_mindir.py \
        --model path/to/encoder.mindir --device npu --profile
"""

from __future__ import annotations

import argparse
import os
import platform
import shlex
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


FEATURE_INPUT_NAMES = {"x", "features", "feature", "feats"}
LENGTH_INPUT_NAMES = {"x_lens", "lengths", "lens"}
ASCEND_PRECISION_MODES = (
    "enforce_fp32",
    "preferred_fp32",
    "enforce_fp16",
    "enforce_origin",
    "preferred_optimal",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the MindIR encoder model.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Default: 1.")
    parser.add_argument("--feature-dim", type=int, default=80, help="Feature dimension. Default: 80.")
    parser.add_argument("--input-frames", type=int, help="Feature input frame count. Required only when the model time dimension is dynamic.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Encoder chunk size at 50 fps. Default: 16.")
    parser.add_argument("--left-context-frames", type=int, default=64, help="Left context at 50 fps. Default: 64.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of stateful warmup calls. Default: 20.")
    parser.add_argument("--loops", type=int, default=100, help="Number of stateful benchmark calls. Default: 100.")
    parser.add_argument(
        "--device",
        choices=("cpu", "ascend", "npu"),
        default="cpu",
        help="Inference device; npu is an alias for ascend. Default: cpu.",
    )
    parser.add_argument("--device-id", type=int, default=0, help="Ascend device ID. Default: 0.")
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
            "must declare supported shape gears in a backend-compatible config."
        ),
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Collect an Ascend profile by launching this benchmark under msprof.",
    )
    parser.add_argument(
        "--profile-output",
        default=f"profiles/{Path(__file__).stem}",
        help="msprof output directory. Used only with --profile.",
    )
    parser.add_argument(
        "--msprof-path",
        default="msprof",
        help="msprof executable name or path. Used only with --profile.",
    )
    parser.add_argument("--profile-child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--cpu", type=int, default=0, help="Host CPU core to bind on Linux. Default: 0.")
    parser.add_argument("--no-cpu-bind", action="store_true", help="Do not bind the process to a host CPU core.")
    parser.add_argument("--seed", type=int, default=42, help="Random feature seed. Default: 42.")
    return parser.parse_args()


def normalize_device(device: str) -> str:
    return "ascend" if device == "npu" else device


def validate_args(args: argparse.Namespace) -> None:
    args.device = normalize_device(args.device)
    if args.batch_size < 1 or args.feature_dim < 1 or args.chunk_size < 1:
        raise ValueError("batch-size, feature-dim, and chunk-size must be positive.")
    if args.input_frames is not None and args.input_frames < 1:
        raise ValueError("input-frames must be positive when provided.")
    if args.left_context_frames < 0 or args.warmup < 0 or args.loops < 1:
        raise ValueError("left-context-frames and warmup must be non-negative; loops must be positive.")
    if args.threads < 1 or args.cpu < 0 or args.device_id < 0:
        raise ValueError("threads must be positive; cpu and device-id must be non-negative.")
    if args.device == "cpu":
        if args.ascend_precision_mode is not None:
            raise ValueError("--ascend-precision-mode requires --device ascend.")
        if args.ascend_provider != "default":
            raise ValueError("--ascend-provider requires --device ascend.")
        if args.profile:
            raise ValueError("--profile requires --device ascend or --device npu.")
    elif args.enable_fp16:
        raise ValueError(
            "--enable-fp16 is CPU-only; use --ascend-precision-mode for Ascend."
        )
    if args.config_path and not Path(args.config_path).expanduser().is_file():
        raise FileNotFoundError(
            f"MindSpore Lite config file does not exist: {args.config_path}"
        )
    if args.profile and not args.profile_output:
        raise ValueError("--profile-output must not be empty when --profile is enabled.")


def bind_cpu(cpu: int) -> None:
    if platform.system() != "Linux":
        print(f"[warning] CPU affinity is not available on {platform.system()}; use the platform affinity tool if needed.", file=sys.stderr)
        return
    try:
        os.sched_setaffinity(0, {cpu})
        print(f"[info] Bound process to CPU {cpu}.")
    except OSError as error:
        print(f"[warning] Could not bind process to CPU {cpu}: {error}", file=sys.stderr)


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
        return precision_mode
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
        "cpu_precision_mode": cpu_precision_mode,
        "requested_ascend_precision_mode": args.ascend_precision_mode,
        "context_ascend_precision_mode": context_precision,
        "requested_ascend_provider": args.ascend_provider,
        "context_ascend_provider": context_provider,
        "cpu_fallback_enabled": args.device == "ascend",
    }
    return context, metadata


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


def resolve_msprof_executable(value: str) -> str:
    candidate = str(Path(value).expanduser()) if os.sep in value else value
    executable = shutil.which(candidate)
    if executable is None:
        raise FileNotFoundError(
            f"msprof executable was not found: {value}. Source the CANN set_env.sh "
            "script or pass --msprof-path."
        )
    return str(Path(executable).resolve())


def create_msprof_command(
    args: argparse.Namespace,
    model_path: Path,
    config_path: Optional[str],
    argv: Optional[Sequence[str]] = None,
) -> Tuple[List[str], Path]:
    executable = resolve_msprof_executable(args.msprof_path)
    output_dir = Path(args.profile_output).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    child_args = list(sys.argv[1:] if argv is None else argv)
    child_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        *child_args,
        "--model",
        str(model_path),
        "--profile-child",
    ]
    if config_path:
        child_command.extend(
            ["--config-path", str(Path(config_path).expanduser().resolve())]
        )
    application = shlex.join(child_command)
    return [
        executable,
        f"--application={application}",
        f"--output={output_dir}",
    ], output_dir


def run_with_msprof(
    args: argparse.Namespace, model_path: Path, config_path: Optional[str]
) -> int:
    command, output_dir = create_msprof_command(args, model_path, config_path)
    print(f"[info] Launching Ascend profiling with msprof; output={output_dir}")
    print(
        "[warning] Profiling adds overhead; do not use this run as the baseline "
        "latency result.",
        file=sys.stderr,
    )
    completed = subprocess.run(command, check=False)
    if completed.returncode == 0:
        print(f"[info] Ascend profile data written under: {output_dir}")
    return completed.returncode


def is_feature_input(name: str) -> bool:
    return name.lower() in FEATURE_INPUT_NAMES


def is_length_input(name: str) -> bool:
    return name.lower() in LENGTH_INPUT_NAMES


def numpy_dtype(tensor: object) -> np.dtype:
    data_type = str(getattr(tensor, "dtype")).lower()
    if "float16" in data_type:
        return np.float16
    if "float32" in data_type or "float" in data_type:
        return np.float32
    if "float64" in data_type or "double" in data_type:
        return np.float64
    if "int64" in data_type:
        return np.int64
    if "int32" in data_type:
        return np.int32
    if "uint8" in data_type:
        return np.uint8
    if "bool" in data_type:
        return np.bool_
    raise ValueError(f"Unsupported MindSpore Lite tensor type: {getattr(tensor, 'dtype')}")


def positive_dimension(value: object) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return 0
    return dimension if dimension > 0 else 0


def resolve_input_frames(raw_shape: Sequence[int], requested: object) -> int:
    shape = list(raw_shape)
    if len(shape) != 3:
        raise ValueError(f"Feature input must be rank 3, got {shape}.")
    fixed = positive_dimension(shape[1])
    if fixed:
        if requested is not None and requested != fixed:
            raise ValueError(f"Feature input has {fixed} frames, but --input-frames is {requested}.")
        return fixed
    if requested is None:
        raise ValueError("Feature input time dimension is dynamic; provide --input-frames.")
    return int(requested)


def resolve_shape(raw_shape: Sequence[int], name: str, args: argparse.Namespace) -> List[int]:
    shape: List[int] = []
    for index, dimension in enumerate(raw_shape):
        value = positive_dimension(dimension)
        if value:
            shape.append(value)
        elif is_feature_input(name) and index < 3:
            shape.append((args.batch_size, args.input_frames, args.feature_dim)[index])
        elif index == 0:
            shape.append(args.batch_size)
        else:
            raise ValueError(f"Cannot resolve dynamic dimension {index} of input {name}: {list(raw_shape)}")
    return shape


def resize_dynamic_inputs(model: object, inputs: Sequence[object], args: argparse.Namespace) -> List[object]:
    shapes = [resolve_shape(tensor.shape, tensor.name, args) for tensor in inputs]
    if any(list(tensor.shape) != shape for tensor, shape in zip(inputs, shapes)):
        model.resize(inputs, shapes)
        return list(model.get_inputs())
    return list(inputs)


def get_feature_index(inputs: Sequence[object]) -> int:
    indexes = [index for index, tensor in enumerate(inputs) if is_feature_input(tensor.name)]
    if len(indexes) != 1:
        raise ValueError(f"Expected exactly one feature input named x/features/feats, found {[tensor.name for tensor in inputs]}.")
    return indexes[0]


def check_model_layout(inputs: Sequence[object], feature_index: int, args: argparse.Namespace) -> int:
    input_frames = resolve_input_frames(inputs[feature_index].shape, args.input_frames)
    args.input_frames = input_frames
    feature_shape = resolve_shape(inputs[feature_index].shape, inputs[feature_index].name, args)
    expected = [args.batch_size, input_frames, args.feature_dim]
    if feature_shape != expected:
        raise ValueError(f"Feature input shape is {feature_shape}; expected {expected}.")
    return input_frames


def reset_states(inputs: Sequence[object], feature_index: int, args: argparse.Namespace) -> None:
    for index, tensor in enumerate(inputs):
        if index == feature_index:
            continue
        shape = resolve_shape(tensor.shape, tensor.name, args)
        if is_length_input(tensor.name):
            values = np.full(shape, args.input_frames, dtype=numpy_dtype(tensor))
        else:
            values = np.zeros(shape, dtype=numpy_dtype(tensor))
        tensor.set_data_from_numpy(values)


def create_state_mapping(inputs: Sequence[object], outputs: Sequence[object]) -> List[Tuple[int, int]]:
    input_indexes = {tensor.name: index for index, tensor in enumerate(inputs)}
    mapping = []
    for output_index, tensor in enumerate(outputs):
        input_name = tensor.name.removeprefix("new_")
        if tensor.name.startswith("new_") and input_name in input_indexes:
            mapping.append((input_indexes[input_name], output_index))
    if mapping:
        return mapping

    state_indexes = [index for index, tensor in enumerate(inputs) if not is_feature_input(tensor.name) and not is_length_input(tensor.name)]
    state_outputs = outputs[1:]
    if len(state_indexes) != len(state_outputs):
        raise ValueError("Could not map output states by name or output order.")
    for input_index, output_tensor in zip(state_indexes, state_outputs):
        input_tensor = inputs[input_index]
        if list(input_tensor.shape) != list(output_tensor.shape) or input_tensor.dtype != output_tensor.dtype:
            raise ValueError(f"State shape or type mismatch: {output_tensor.name} -> {input_tensor.name}.")
    print("[warning] Output names do not preserve new_* state names; using ordered state mapping.", file=sys.stderr)
    return [(input_index, output_index + 1) for output_index, input_index in enumerate(state_indexes)]


def set_features(inputs: Sequence[object], feature_index: int, features: np.ndarray) -> None:
    tensor = inputs[feature_index]
    tensor.set_data_from_numpy(np.ascontiguousarray(features, dtype=numpy_dtype(tensor)))


def update_states(inputs: Sequence[object], outputs: Sequence[object], state_mapping: Sequence[Tuple[int, int]]) -> None:
    for input_index, output_index in state_mapping:
        values = np.array(outputs[output_index].get_data_to_numpy(), copy=True)
        inputs[input_index].set_data_from_numpy(np.ascontiguousarray(values, dtype=numpy_dtype(inputs[input_index])))


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


def print_summary(latencies_ms: List[float], args: argparse.Namespace) -> None:
    summary = summarize(latencies_ms)
    fill_steps = (args.left_context_frames + args.chunk_size - 1) // args.chunk_size
    steady = latencies_ms[fill_steps:]
    print("\n[result] Streaming inference latency (ms)")
    for name in ("mean", "std", "min", "p50", "p90", "p95", "p99", "max"):
        print(f"  {name:>4}: {summary[name]:.3f}")
    print(f"  Calls: {len(latencies_ms)}")
    print(f"  Step duration: {args.chunk_size * 20} ms")
    print(f"  RTF (p50): {summary['p50'] / (args.chunk_size * 20):.6f}")
    print(f"  Cache-fill calls: {fill_steps}")
    if steady:
        print(f"  Steady-state p50 after cache fill: {np.percentile(steady, 50):.3f} ms")


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        config_path = (
            str(Path(args.config_path).expanduser().resolve())
            if args.config_path
            else None
        )
        if args.profile and not args.profile_child:
            raise SystemExit(run_with_msprof(args, model_path, config_path))
        if not args.no_cpu_bind:
            bind_cpu(args.cpu)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
        import mindspore_lite as mslite

        context, context_metadata = create_context(mslite, args)
        model = build_model(mslite, model_path, context, config_path)
        inputs = list(model.get_inputs())
        feature_index = get_feature_index(inputs)
        args.input_frames = resolve_input_frames(inputs[feature_index].shape, args.input_frames)
        inputs = resize_dynamic_inputs(model, inputs, args)
        feature_index = get_feature_index(inputs)
        input_frames = check_model_layout(inputs, feature_index, args)
        print(f"[info] Model: {model_path}")
        print("[info] Model inputs:")
        for tensor in inputs:
            print(f"  {tensor.name}: shape={tensor.shape}, type={tensor.dtype}")
        print(
            f"[info] device={args.device}, input_frames={input_frames}, "
            f"chunk_size={args.chunk_size}, left_context_frames={args.left_context_frames}, "
            f"warmup={args.warmup}, loops={args.loops}, threads={args.threads}, "
            f"enable_fp16={args.enable_fp16}"
        )
        if args.device == "ascend":
            print(
                f"[info] device_id={context_metadata['device_id']}, "
                f"ascend_precision_mode={context_metadata['context_ascend_precision_mode']}, "
                f"ascend_provider={context_metadata['context_ascend_provider']}"
            )
            print(
                "[warning] An Ascend target may retain CPU fallback for some nodes; "
                "successful execution alone does not prove full NPU offload.",
                file=sys.stderr,
            )
        if args.profile:
            print(
                f"[info] Ascend profiling is active; output root="
                f"{Path(args.profile_output).expanduser().resolve()}"
            )

        shift_frames = 2 * args.chunk_size
        total_frames = input_frames + (max(args.warmup, args.loops) - 1) * shift_frames
        features = np.random.default_rng(args.seed).standard_normal((total_frames, args.feature_dim), dtype=np.float32)

        reset_states(inputs, feature_index, args)
        set_features(inputs, feature_index, np.zeros((args.batch_size, input_frames, args.feature_dim), dtype=np.float32))
        outputs = model.predict(inputs)
        state_mapping = create_state_mapping(inputs, outputs)

        reset_states(inputs, feature_index, args)
        for step in range(args.warmup):
            offset = step * shift_frames
            set_features(inputs, feature_index, features[offset:offset + input_frames][None])
            outputs = model.predict(inputs)
            update_states(inputs, outputs, state_mapping)

        reset_states(inputs, feature_index, args)
        latencies_ms = []
        for step in range(args.loops):
            offset = step * shift_frames
            set_features(inputs, feature_index, features[offset:offset + input_frames][None])
            started = time.perf_counter()
            outputs = model.predict(inputs)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            update_states(inputs, outputs, state_mapping)

        print_summary(latencies_ms, args)
    except (FileNotFoundError, ImportError, OSError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
