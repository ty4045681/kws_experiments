#!/usr/bin/env python3
"""Generate fixed streaming Zipformer encoder inputs for native CLI benchmarks.

Example:
    python scripts/generate_zipformer_streaming_fixtures.py \
        --onnx-model path/to/encoder.onnx \
        --mindir-model path/to/encoder.mindir \
        --output-dir fixtures/zipformer
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


FEATURE_INPUT_NAMES = {"x", "features", "feature", "feats"}
LENGTH_INPUT_NAMES = {"x_lens", "lengths", "lens"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-model", help="Path to the ONNX streaming encoder.")
    parser.add_argument("--mindir-model", help="Path to the MindIR streaming encoder.")
    parser.add_argument("--output-dir", required=True, help="Directory for generated fixtures.")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size. Default: 1.")
    parser.add_argument("--feature-dim", type=int, default=80, help="Feature dimension. Default: 80.")
    parser.add_argument("--chunk-size", type=int, default=16, help="Encoder chunk size at 50 fps. Default: 16.")
    parser.add_argument("--left-context-frames", type=int, default=64, help="Left context at 50 fps. Default: 64.")
    parser.add_argument("--threads", type=int, default=1, help="Runtime CPU thread count. Default: 1.")
    parser.add_argument("--seed", type=int, default=42, help="Random feature seed. Default: 42.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.onnx_model and not args.mindir_model:
        raise ValueError("At least one of --onnx-model or --mindir-model is required.")
    for model in (args.onnx_model, args.mindir_model):
        if model and not Path(model).is_file():
            raise FileNotFoundError(f"Model file does not exist: {model}")
    if args.batch_size < 1 or args.feature_dim < 1 or args.chunk_size < 1 or args.threads < 1:
        raise ValueError("batch-size, feature-dim, chunk-size, and threads must be positive.")
    if args.left_context_frames < 0:
        raise ValueError("left-context-frames must be non-negative.")


def is_feature_input(name: str) -> bool:
    return name.lower() in FEATURE_INPUT_NAMES


def is_length_input(name: str) -> bool:
    return name.lower() in LENGTH_INPUT_NAMES


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "tensor"


def case_name(step: int, fill_steps: int) -> str:
    if step == 0:
        return "step_00_initial"
    if step == fill_steps:
        return f"step_{step:02d}_steady"
    return f"step_{step:02d}"


def resolve_shape(raw_shape: Iterable[Any], name: str, args: argparse.Namespace) -> List[int]:
    result = []
    for index, dimension in enumerate(raw_shape):
        try:
            value = int(dimension)
        except (TypeError, ValueError):
            value = 0
        if value > 0:
            result.append(value)
        elif is_feature_input(name) and index < 3:
            result.append((args.batch_size, 2 * args.chunk_size + 7, args.feature_dim)[index])
        elif index == 0:
            result.append(args.batch_size)
        else:
            raise ValueError(f"Cannot resolve dynamic dimension {index} of input {name}: {list(raw_shape)}")
    return result


def save_raw(array: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.ascontiguousarray(array).tofile(path)


def input_entry(index: int, name: str, array: np.ndarray, path: Path, root: Path) -> Dict[str, Any]:
    contiguous = np.ascontiguousarray(array)
    return {
        "index": index,
        "name": name,
        "dtype": str(contiguous.dtype),
        "dtype_str": contiguous.dtype.str,
        "shape": list(contiguous.shape),
        "element_count": int(contiguous.size),
        "byte_size": int(contiguous.nbytes),
        "file": str(path.relative_to(root)),
    }


def prepared_features(args: argparse.Namespace, fixture_count: int) -> List[np.ndarray]:
    input_frames = 2 * args.chunk_size + 7
    shift_frames = 2 * args.chunk_size
    total_frames = input_frames + (fixture_count - 1) * shift_frames
    stream = np.random.default_rng(args.seed).standard_normal((total_frames, args.feature_dim), dtype=np.float32)
    return [np.ascontiguousarray(stream[step * shift_frames:step * shift_frames + input_frames][None]) for step in range(fixture_count)]


def write_feature_windows(output_dir: Path, features: Sequence[np.ndarray], args: argparse.Namespace, fill_steps: int) -> List[str]:
    paths = []
    for step, window in enumerate(features):
        path = output_dir / "features" / f"{case_name(step, fill_steps)}.npy"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, window)
        paths.append(str(path.relative_to(output_dir)))
    return paths


def onnx_dtype(onnx_type: str) -> np.dtype:
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


def onnx_initial_feed(session: Any, args: argparse.Namespace) -> Dict[str, np.ndarray]:
    feed = {}
    for meta in session.get_inputs():
        if is_feature_input(meta.name):
            continue
        shape = resolve_shape(meta.shape, meta.name, args)
        dtype = onnx_dtype(meta.type)
        feed[meta.name] = np.full(shape, 2 * args.chunk_size + 7, dtype=dtype) if is_length_input(meta.name) else np.zeros(shape, dtype=dtype)
    return feed


def onnx_state_mapping(session: Any) -> List[Tuple[str, int]]:
    inputs = session.get_inputs()
    outputs = session.get_outputs()
    input_names = {meta.name for meta in inputs}
    mapping = [(meta.name.removeprefix("new_"), index) for index, meta in enumerate(outputs) if meta.name.startswith("new_") and meta.name.removeprefix("new_") in input_names]
    if mapping:
        return mapping
    state_inputs = [meta for meta in inputs if not is_feature_input(meta.name) and not is_length_input(meta.name)]
    state_outputs = outputs[1:]
    if len(state_inputs) != len(state_outputs):
        raise ValueError("Could not map ONNX output states by name or output order.")
    for input_meta, output_meta in zip(state_inputs, state_outputs):
        if list(input_meta.shape) != list(output_meta.shape) or input_meta.type != output_meta.type:
            raise ValueError(f"ONNX state shape or type mismatch: {output_meta.name} -> {input_meta.name}.")
    print("[warning] ONNX output names do not preserve new_* state names; using ordered state mapping.", file=sys.stderr)
    return [(meta.name, index + 1) for index, meta in enumerate(state_inputs)]


def check_onnx_layout(session: Any, args: argparse.Namespace) -> None:
    feature_inputs = [meta for meta in session.get_inputs() if is_feature_input(meta.name)]
    if len(feature_inputs) != 1:
        raise ValueError(f"Expected one ONNX feature input, found {[meta.name for meta in feature_inputs]}.")
    shape = resolve_shape(feature_inputs[0].shape, feature_inputs[0].name, args)
    expected = [args.batch_size, 2 * args.chunk_size + 7, args.feature_dim]
    if shape != expected:
        raise ValueError(f"ONNX feature input shape is {shape}; expected {expected}.")


def processed_lens(feed: Dict[str, np.ndarray]) -> Optional[List[int]]:
    for name, values in feed.items():
        if "processed_lens" in name.lower():
            return np.asarray(values).astype(np.int64).reshape(-1).tolist()
    return None


def write_onnx_fixture(output_dir: Path, session: Any, feed: Dict[str, np.ndarray], feature_window: np.ndarray, step: int, fill_steps: int) -> Dict[str, Any]:
    directory = output_dir / "onnx" / case_name(step, fill_steps)
    inputs = []
    for index, meta in enumerate(session.get_inputs()):
        array = feature_window.astype(onnx_dtype(meta.type), copy=False) if is_feature_input(meta.name) else feed[meta.name]
        path = directory / f"input_{index:02d}_{safe_name(meta.name)}.bin"
        save_raw(array, path)
        inputs.append(input_entry(index, meta.name, array, path, output_dir))
    return {"name": case_name(step, fill_steps), "step": step, "processed_lens": processed_lens(feed), "inputs": inputs}


def generate_onnx(output_dir: Path, features: Sequence[np.ndarray], args: argparse.Namespace, fill_steps: int) -> Dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError as error:
        raise ImportError("Generating ONNX fixtures requires onnxruntime.") from error
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = args.threads
    options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(args.onnx_model, sess_options=options, providers=["CPUExecutionProvider"])
    check_onnx_layout(session, args)
    mapping = onnx_state_mapping(session)
    output_names = [meta.name for meta in session.get_outputs()]
    feed = onnx_initial_feed(session, args)
    fixtures = []
    for step, window in enumerate(features):
        fixtures.append(write_onnx_fixture(output_dir, session, feed, window, step, fill_steps))
        if step == fill_steps:
            continue
        feature_name = next(meta.name for meta in session.get_inputs() if is_feature_input(meta.name))
        feature_meta = next(meta for meta in session.get_inputs() if meta.name == feature_name)
        feed[feature_name] = window.astype(onnx_dtype(feature_meta.type), copy=False)
        outputs = session.run(output_names, feed)
        feed.pop(feature_name)
        for input_name, output_index in mapping:
            feed[input_name] = np.array(outputs[output_index], copy=True)
    return {
        "backend": "onnx",
        "model": str(Path(args.onnx_model).resolve()),
        "input_order": [meta.name for meta in session.get_inputs()],
        "fixtures": fixtures,
    }


def mindir_dtype(tensor: object) -> np.dtype:
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


def mindir_resize(model: object, inputs: Sequence[object], args: argparse.Namespace) -> List[object]:
    shapes = [resolve_shape(tensor.shape, tensor.name, args) for tensor in inputs]
    if any(list(tensor.shape) != shape for tensor, shape in zip(inputs, shapes)):
        model.resize(inputs, shapes)
        return list(model.get_inputs())
    return list(inputs)


def mindir_feature_index(inputs: Sequence[object]) -> int:
    indexes = [index for index, tensor in enumerate(inputs) if is_feature_input(tensor.name)]
    if len(indexes) != 1:
        raise ValueError(f"Expected one MindIR feature input, found {[tensor.name for tensor in inputs]}.")
    return indexes[0]


def check_mindir_layout(inputs: Sequence[object], feature_index: int, args: argparse.Namespace) -> None:
    shape = resolve_shape(inputs[feature_index].shape, inputs[feature_index].name, args)
    expected = [args.batch_size, 2 * args.chunk_size + 7, args.feature_dim]
    if shape != expected:
        raise ValueError(f"MindIR feature input shape is {shape}; expected {expected}.")


def mindir_reset_states(inputs: Sequence[object], feature_index: int, args: argparse.Namespace) -> None:
    for index, tensor in enumerate(inputs):
        if index == feature_index:
            continue
        shape = resolve_shape(tensor.shape, tensor.name, args)
        values = np.full(shape, 2 * args.chunk_size + 7, dtype=mindir_dtype(tensor)) if is_length_input(tensor.name) else np.zeros(shape, dtype=mindir_dtype(tensor))
        tensor.set_data_from_numpy(values)


def mindir_state_mapping(inputs: Sequence[object], outputs: Sequence[object]) -> List[Tuple[int, int]]:
    indexes = {tensor.name: index for index, tensor in enumerate(inputs)}
    mapping = [(indexes[tensor.name.removeprefix("new_")], output_index) for output_index, tensor in enumerate(outputs) if tensor.name.startswith("new_") and tensor.name.removeprefix("new_") in indexes]
    if mapping:
        return mapping
    state_indexes = [index for index, tensor in enumerate(inputs) if not is_feature_input(tensor.name) and not is_length_input(tensor.name)]
    state_outputs = outputs[1:]
    if len(state_indexes) != len(state_outputs):
        raise ValueError("Could not map MindIR output states by name or output order.")
    for input_index, output_tensor in zip(state_indexes, state_outputs):
        input_tensor = inputs[input_index]
        if list(input_tensor.shape) != list(output_tensor.shape) or input_tensor.dtype != output_tensor.dtype:
            raise ValueError(f"MindIR state shape or type mismatch: {output_tensor.name} -> {input_tensor.name}.")
    print("[warning] MindIR output names do not preserve new_* state names; using ordered state mapping.", file=sys.stderr)
    return [(input_index, output_index + 1) for output_index, input_index in enumerate(state_indexes)]


def mindir_inputs_to_arrays(inputs: Sequence[object]) -> Dict[str, np.ndarray]:
    return {tensor.name: np.array(tensor.get_data_to_numpy(), copy=True) for tensor in inputs}


def write_mindir_fixture(output_dir: Path, inputs: Sequence[object], feature_index: int, feature_window: np.ndarray, step: int, fill_steps: int) -> Dict[str, Any]:
    directory = output_dir / "mindir" / case_name(step, fill_steps)
    entries = []
    for index, tensor in enumerate(inputs):
        array = feature_window.astype(mindir_dtype(tensor), copy=False) if index == feature_index else np.array(tensor.get_data_to_numpy(), copy=True)
        path = directory / f"input_{index:02d}_{safe_name(tensor.name)}.bin"
        save_raw(array, path)
        entries.append(input_entry(index, tensor.name, array, path, output_dir))
    state_arrays = mindir_inputs_to_arrays(inputs)
    return {"name": case_name(step, fill_steps), "step": step, "processed_lens": processed_lens(state_arrays), "inputs": entries}


def generate_mindir(output_dir: Path, features: Sequence[np.ndarray], args: argparse.Namespace, fill_steps: int) -> Dict[str, Any]:
    try:
        import mindspore_lite as mslite
    except ImportError as error:
        raise ImportError("Generating MindIR fixtures requires mindspore_lite.") from error
    context = mslite.Context()
    context.target = ["cpu"]
    context.cpu.thread_num = args.threads
    context.cpu.enable_fp16 = False
    model = mslite.Model()
    model.build_from_file(args.mindir_model, mslite.ModelType.MINDIR, context)
    inputs = mindir_resize(model, model.get_inputs(), args)
    feature_index = mindir_feature_index(inputs)
    check_mindir_layout(inputs, feature_index, args)
    mindir_reset_states(inputs, feature_index, args)
    inputs[feature_index].set_data_from_numpy(np.zeros(resolve_shape(inputs[feature_index].shape, inputs[feature_index].name, args), dtype=mindir_dtype(inputs[feature_index])))
    mapping = mindir_state_mapping(inputs, model.predict(inputs))
    mindir_reset_states(inputs, feature_index, args)
    fixtures = []
    for step, window in enumerate(features):
        fixtures.append(write_mindir_fixture(output_dir, inputs, feature_index, window, step, fill_steps))
        if step == fill_steps:
            continue
        inputs[feature_index].set_data_from_numpy(window.astype(mindir_dtype(inputs[feature_index]), copy=False))
        outputs = model.predict(inputs)
        for input_index, output_index in mapping:
            values = np.array(outputs[output_index].get_data_to_numpy(), copy=True)
            inputs[input_index].set_data_from_numpy(np.ascontiguousarray(values, dtype=mindir_dtype(inputs[input_index])))
    return {
        "backend": "mindir",
        "model": str(Path(args.mindir_model).resolve()),
        "input_order": [tensor.name for tensor in inputs],
        "fixtures": fixtures,
    }


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {path}. Use --overwrite to write into it.")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        output_dir = Path(args.output_dir).resolve()
        prepare_output_dir(output_dir, args.overwrite)
        fill_steps = math.ceil(args.left_context_frames / args.chunk_size)
        fixture_count = fill_steps + 1
        features = prepared_features(args, fixture_count)
        feature_files = write_feature_windows(output_dir, features, args, fill_steps)
        print(f"[info] Generating {fixture_count} fixtures: initial plus {fill_steps} state-advance steps.")
        backends = {}
        if args.onnx_model:
            backends["onnx"] = generate_onnx(output_dir, features, args, fill_steps)
            print(f"[info] Wrote ONNX fixtures to {output_dir / 'onnx'}.")
        if args.mindir_model:
            backends["mindir"] = generate_mindir(output_dir, features, args, fill_steps)
            print(f"[info] Wrote MindIR fixtures to {output_dir / 'mindir'}.")
        manifest = {
            "format_version": 1,
            "generator": Path(__file__).name,
            "configuration": {
                "batch_size": args.batch_size,
                "feature_dim": args.feature_dim,
                "chunk_size": args.chunk_size,
                "left_context_frames": args.left_context_frames,
                "state_advance_steps": fill_steps,
                "fixture_count": fixture_count,
                "seed": args.seed,
            },
            "features": feature_files,
            "backends": backends,
        }
        (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"[info] Wrote manifest: {output_dir / 'manifest.json'}.")
    except (FileNotFoundError, ImportError, RuntimeError, ValueError) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
