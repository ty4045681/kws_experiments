#!/usr/bin/env python3
"""Benchmark a stateful streaming Zipformer MindIR encoder with MindSpore Lite.

Example:
    python scripts/bench_zipformer_streaming_mindir.py \
        --model path/to/encoder.mindir
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


FEATURE_INPUT_NAMES = {"x", "features", "feature", "feats"}
LENGTH_INPUT_NAMES = {"x_lens", "lengths", "lens"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Path to the MindIR encoder model.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Default: 1.")
    parser.add_argument("--feature-dim", type=int, default=80, help="Feature dimension. Default: 80.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Encoder chunk size at 50 fps. Default: 16.")
    parser.add_argument("--left-context-frames", type=int, default=64, help="Left context at 50 fps. Default: 64.")
    parser.add_argument("--warmup", type=int, default=20, help="Number of stateful warmup calls. Default: 20.")
    parser.add_argument("--loops", type=int, default=100, help="Number of stateful benchmark calls. Default: 100.")
    parser.add_argument("--threads", type=int, default=1, help="MindSpore Lite CPU thread count. Default: 1.")
    parser.add_argument("--cpu", type=int, default=0, help="CPU core to bind on Linux. Default: 0.")
    parser.add_argument("--no-cpu-bind", action="store_true", help="Do not bind the process to a CPU core.")
    parser.add_argument("--enable-fp16", action="store_true", help="Enable MindSpore Lite CPU FP16 mode.")
    parser.add_argument("--seed", type=int, default=42, help="Random feature seed. Default: 42.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.feature_dim < 1 or args.chunk_size < 1:
        raise ValueError("batch-size, feature-dim, and chunk-size must be positive.")
    if args.left_context_frames < 0 or args.warmup < 0 or args.loops < 1:
        raise ValueError("left-context-frames and warmup must be non-negative; loops must be positive.")
    if args.threads < 1 or args.cpu < 0:
        raise ValueError("threads must be positive and cpu must be non-negative.")


def bind_cpu(cpu: int) -> None:
    if platform.system() != "Linux":
        print(f"[warning] CPU affinity is not available on {platform.system()}; use the platform affinity tool if needed.", file=sys.stderr)
        return
    try:
        os.sched_setaffinity(0, {cpu})
        print(f"[info] Bound process to CPU {cpu}.")
    except OSError as error:
        print(f"[warning] Could not bind process to CPU {cpu}: {error}", file=sys.stderr)


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


def resolve_shape(raw_shape: Sequence[int], name: str, args: argparse.Namespace) -> List[int]:
    shape: List[int] = []
    for index, dimension in enumerate(raw_shape):
        value = int(dimension)
        if value > 0:
            shape.append(value)
        elif is_feature_input(name):
            shape.append((args.batch_size, 2 * args.chunk_size + 7, args.feature_dim)[index] if index < 3 else 1)
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


def check_model_layout(inputs: Sequence[object], feature_index: int, args: argparse.Namespace) -> None:
    feature_shape = resolve_shape(inputs[feature_index].shape, inputs[feature_index].name, args)
    expected_frames = 2 * args.chunk_size + 7
    if len(feature_shape) != 3:
        raise ValueError(f"Feature input {inputs[feature_index].name} must be rank 3, got {feature_shape}.")
    if feature_shape[1] != expected_frames:
        raise ValueError(
            f"Feature input has {feature_shape[1]} frames, but chunk-size {args.chunk_size} requires {expected_frames}. "
            "Use the chunk size used to export this model."
        )
    if feature_shape[2] != args.feature_dim:
        raise ValueError(f"Feature input has dimension {feature_shape[2]}, but --feature-dim is {args.feature_dim}.")


def reset_states(inputs: Sequence[object], feature_index: int, args: argparse.Namespace) -> None:
    for index, tensor in enumerate(inputs):
        if index == feature_index:
            continue
        shape = resolve_shape(tensor.shape, tensor.name, args)
        if is_length_input(tensor.name):
            values = np.full(shape, 2 * args.chunk_size + 7, dtype=numpy_dtype(tensor))
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
        if not Path(args.model).is_file():
            raise FileNotFoundError(f"Model file does not exist: {args.model}")
        if not args.no_cpu_bind:
            bind_cpu(args.cpu)
        os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
        os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
        import mindspore_lite as mslite

        context = mslite.Context()
        context.target = ["cpu"]
        context.cpu.thread_num = args.threads
        context.cpu.enable_fp16 = args.enable_fp16
        model = mslite.Model()
        model.build_from_file(args.model, mslite.ModelType.MINDIR, context)
        inputs = resize_dynamic_inputs(model, model.get_inputs(), args)
        feature_index = get_feature_index(inputs)
        check_model_layout(inputs, feature_index, args)
        print("[info] Model inputs:")
        for tensor in inputs:
            print(f"  {tensor.name}: shape={tensor.shape}, type={tensor.dtype}")
        print(f"[info] chunk_size={args.chunk_size}, left_context_frames={args.left_context_frames}, warmup={args.warmup}, loops={args.loops}, threads={args.threads}, enable_fp16={args.enable_fp16}")

        input_frames = 2 * args.chunk_size + 7
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
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
