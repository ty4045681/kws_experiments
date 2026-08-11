#!/usr/bin/env python3
"""Benchmark a stateful streaming Zipformer ONNX encoder on CPU.

Example:
    python scripts/bench_zipformer_streaming_onnx.py \
        --model path/to/encoder.onnx
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


FEATURE_INPUT_NAMES = {"x", "features", "feature", "feats"}
LENGTH_INPUT_NAMES = {"x_lens", "lengths", "lens"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the ONNX encoder model.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Default: 1.")
    parser.add_argument("--feature-dim", type=int, default=80, help="Feature dimension. Default: 80.")
    parser.add_argument("--input-frames", type=int, help="Feature input frame count. Required only when the model time dimension is dynamic.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Encoder chunk size at 50 fps. Default: 16.")
    parser.add_argument("--left-context-frames", type=int, default=64, help="Left context at 50 fps. Default: 64.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of stateful warmup calls. Default: 20.")
    parser.add_argument("--loops", type=int, default=100, help="Number of stateful benchmark calls. Default: 100.")
    parser.add_argument("--threads", type=int, default=1, help="ONNX Runtime intra-op thread count. Default: 1.")
    parser.add_argument("--inter-op-threads", type=int, default=1, help="ONNX Runtime inter-op thread count. Default: 1.")
    parser.add_argument("--profile", action="store_true", help="Enable ONNX Runtime profiling. Disabled by default.")
    parser.add_argument(
        "--profile-prefix",
        default=f"profiles/{Path(__file__).stem}",
        help="Profile file prefix; ONNX Runtime appends a timestamp. Used only with --profile.",
    )
    parser.add_argument("--cpu", type=int, default=0, help="CPU core to bind on Linux. Default: 0.")
    parser.add_argument("--no-cpu-bind", action="store_true", help="Do not bind the process to a CPU core.")
    parser.add_argument(
        "--disable-optimizer",
        action="append",
        default=[],
        metavar="OPTIMIZER_NAME",
        help="Disable a named ONNX Runtime graph optimizer. Repeatable; e.g. --disable-optimizer NhwcTransformer.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random feature seed. Default: 42.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.feature_dim < 1 or args.chunk_size < 1:
        raise ValueError("batch-size, feature-dim, and chunk-size must be positive.")
    if args.input_frames is not None and args.input_frames < 1:
        raise ValueError("input-frames must be positive when provided.")
    if args.left_context_frames < 0 or args.warmup < 0 or args.loops < 1:
        raise ValueError("left-context-frames and warmup must be non-negative; loops must be positive.")
    if args.threads < 1 or args.inter_op_threads < 1 or args.cpu < 0:
        raise ValueError("Thread counts must be positive and cpu must be non-negative.")


def bind_cpu(cpu: int) -> None:
    if platform.system() != "Linux":
        print(f"[warning] CPU affinity is not available on {platform.system()}; use the platform affinity tool if needed.", file=sys.stderr)
        return
    try:
        os.sched_setaffinity(0, {cpu})
        print(f"[info] Bound process to CPU {cpu}.")
    except OSError as error:
        print(f"[warning] Could not bind process to CPU {cpu}: {error}", file=sys.stderr)


def configure_profiling(options: Any, enabled: bool, prefix: str) -> None:
    if not enabled:
        return
    profile_prefix = Path(prefix).expanduser().resolve()
    profile_prefix.parent.mkdir(parents=True, exist_ok=True)
    options.enable_profiling = True
    options.profile_file_prefix = str(profile_prefix)
    print(f"[info] ONNX Runtime profiling enabled; prefix={profile_prefix}")
    print("[warning] Profiling adds overhead; do not use this run as the baseline latency result.", file=sys.stderr)


def finish_profiling(session: Any, enabled: bool) -> Optional[str]:
    if not enabled:
        return None
    profile_path = str(Path(session.end_profiling()).resolve())
    print(f"[info] Wrote ONNX Runtime profile: {profile_path}")
    return profile_path


def numpy_dtype(onnx_type: str) -> np.dtype:
    types = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": np.bool_,
    }
    try:
        return types[onnx_type]
    except KeyError as error:
        raise ValueError(f"Unsupported ONNX input type: {onnx_type}") from error


def is_feature_input(name: str) -> bool:
    return name.lower() in FEATURE_INPUT_NAMES


def is_length_input(name: str) -> bool:
    return name.lower() in LENGTH_INPUT_NAMES


def positive_dimension(value: Any) -> int:
    try:
        dimension = int(value)
    except (TypeError, ValueError):
        return 0
    return dimension if dimension > 0 else 0


def resolve_input_frames(raw_shape: Iterable[Any], requested: Any) -> int:
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
    return requested


def resolve_shape(raw_shape: Iterable[Any], name: str, args: argparse.Namespace) -> List[int]:
    dimensions = list(raw_shape)
    shape: List[int] = []
    for index, dimension in enumerate(dimensions):
        value = positive_dimension(dimension)
        if value:
            shape.append(value)
        elif is_feature_input(name) and index < 3:
            shape.append((args.batch_size, args.input_frames, args.feature_dim)[index])
        elif index == 0:
            shape.append(args.batch_size)
        else:
            raise ValueError(f"Cannot resolve dynamic dimension {index} of input {name}: {dimensions}")
    return shape


def create_initial_feed(session: Any, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    feed: Dict[str, np.ndarray] = {}
    for input_meta in session.get_inputs():
        name = input_meta.name
        shape = resolve_shape(input_meta.shape, name, args)
        dtype = numpy_dtype(input_meta.type)
        if is_feature_input(name):
            continue
        if is_length_input(name):
            feed[name] = np.full(shape, args.input_frames, dtype=dtype)
        else:
            feed[name] = np.zeros(shape, dtype=dtype)
    return feed


def create_state_mapping(session: Any) -> List[Tuple[str, int]]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_names = {meta.name for meta in inputs}
    mapping = []
    for index, output_meta in enumerate(outputs):
        input_name = output_meta.name.removeprefix("new_")
        if output_meta.name.startswith("new_") and input_name in input_names:
            mapping.append((input_name, index))
    if mapping:
        return mapping

    state_inputs = [meta for meta in inputs if not is_feature_input(meta.name) and not is_length_input(meta.name)]
    state_outputs = outputs[1:]
    if len(state_inputs) != len(state_outputs):
        raise ValueError("Could not map output states by name or output order.")
    for input_meta, output_meta in zip(state_inputs, state_outputs):
        if list(input_meta.shape) != list(output_meta.shape) or input_meta.type != output_meta.type:
            raise ValueError(f"State shape or type mismatch: {output_meta.name} -> {input_meta.name}.")
    print("[warning] Output names do not preserve new_* state names; using ordered state mapping.", file=sys.stderr)
    return [(input_meta.name, index + 1) for index, input_meta in enumerate(state_inputs)]


def check_model_layout(session: Any, args: argparse.Namespace) -> int:
    feature_inputs = [meta for meta in session.get_inputs() if is_feature_input(meta.name)]
    if len(feature_inputs) != 1:
        raise ValueError(f"Expected exactly one feature input named x/features/feats, found {[meta.name for meta in feature_inputs]}.")
    input_frames = resolve_input_frames(feature_inputs[0].shape, args.input_frames)
    args.input_frames = input_frames
    feature_shape = resolve_shape(feature_inputs[0].shape, feature_inputs[0].name, args)
    expected = [args.batch_size, input_frames, args.feature_dim]
    if feature_shape != expected:
        raise ValueError(f"Feature input shape is {feature_shape}; expected {expected}.")
    return input_frames


def reset_feed(session: Any, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    return create_initial_feed(session, args)


def apply_features(feed: Dict[str, np.ndarray], session: Any, features: np.ndarray) -> None:
    feature_meta = next(meta for meta in session.get_inputs() if is_feature_input(meta.name))
    feed[feature_meta.name] = np.ascontiguousarray(features, dtype=numpy_dtype(feature_meta.type))


def run_step(session: Any, output_names: List[str], state_mapping: List[Tuple[str, int]], feed: Dict[str, np.ndarray], features: np.ndarray) -> None:
    apply_features(feed, session, features)
    outputs = session.run(output_names, feed)
    for input_name, output_index in state_mapping:
        feed[input_name] = outputs[output_index]


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
        if not Path(args.model).is_file():
            raise FileNotFoundError(f"Model file does not exist: {args.model}")
        if not args.no_cpu_bind:
            bind_cpu(args.cpu)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
        import onnxruntime as ort

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        options.intra_op_num_threads = args.threads
        options.inter_op_num_threads = args.inter_op_threads
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        if args.disable_optimizer:
            # 1.28.0 的 SessionOptions 没有 disabled_optimizers 属性；
            # 通过 session config key 传入，1.19+ 均支持（分号分隔）。
            options.add_session_config_entry(
                "optimization.disable_specified_optimizers",
                ";".join(args.disable_optimizer),
            )
        configure_profiling(options, args.profile, args.profile_prefix)
        session = ort.InferenceSession(args.model, sess_options=options, providers=["CPUExecutionProvider"])
        if args.disable_optimizer:
            print(f"[info] Disabled optimizers: {args.disable_optimizer}")
        input_frames = check_model_layout(session, args)
        state_mapping = create_state_mapping(session)
        output_names = [meta.name for meta in session.get_outputs()]
        print("[info] Model inputs:")
        for meta in session.get_inputs():
            print(f"  {meta.name}: shape={meta.shape}, type={meta.type}")
        print(f"[info] input_frames={input_frames}, chunk_size={args.chunk_size}, left_context_frames={args.left_context_frames}, warmup={args.warmup}, loops={args.loops}, threads={args.threads}")

        shift_frames = 2 * args.chunk_size
        total_frames = input_frames + (max(args.warmup, args.loops) - 1) * shift_frames
        features = np.random.default_rng(args.seed).standard_normal((total_frames, args.feature_dim), dtype=np.float32)

        feed = reset_feed(session, args)
        for step in range(args.warmup):
            offset = step * shift_frames
            run_step(session, output_names, state_mapping, feed, features[offset:offset + input_frames][None])

        feed = reset_feed(session, args)
        latencies_ms = []
        for step in range(args.loops):
            offset = step * shift_frames
            window = features[offset:offset + input_frames][None]
            apply_features(feed, session, window)
            started = time.perf_counter()
            outputs = session.run(output_names, feed)
            latencies_ms.append((time.perf_counter() - started) * 1000.0)
            for input_name, output_index in state_mapping:
                feed[input_name] = outputs[output_index]

        finish_profiling(session, args.profile)
        print_summary(latencies_ms, args)
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
