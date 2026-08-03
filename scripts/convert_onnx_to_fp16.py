#!/usr/bin/env python3
r"""Convert a valid FP32 ONNX model to FP16 and validate the result.

The converter uses ``onnxruntime.transformers.float16`` so existing
``Cast(to=FLOAT)`` nodes are updated consistently with FP16 value metadata.
Both the source and converted models are checked with ONNX full type/shape
inference. The converted model is written through a temporary file and moved
to the requested destination only after all enabled checks pass.

Example:
    uv run --with onnx==1.22.0 --with onnxruntime==1.28.0 \
      python scripts/convert_onnx_to_fp16.py \
      --input-model path/to/encoder.onnx \
      --output-model path/to/encoder.fp16.onnx \
      --runtime-provider CPUExecutionProvider

Use ``--keep-io-types`` when the deployment interface must retain FP32 graph
inputs and outputs while internal floating-point computation uses FP16.
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Optional


MIN_FLOAT16_SUBNORMAL = 5.96e-08
MAX_FLOAT16_FINITE = 65504.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input-model",
        required=True,
        help="Path to the valid source FP32 ONNX model.",
    )
    parser.add_argument(
        "--output-model",
        required=True,
        help="Path for the converted FP16 ONNX model.",
    )
    parser.add_argument(
        "--keep-io-types",
        action="store_true",
        help="Keep floating-point graph inputs and outputs as FP32.",
    )
    parser.add_argument(
        "--disable-shape-infer",
        action="store_true",
        help=(
            "Skip the converter's preliminary shape inference. The mandatory "
            "post-conversion ONNX full check is still performed."
        ),
    )
    parser.add_argument(
        "--min-positive-val",
        type=float,
        default=MIN_FLOAT16_SUBNORMAL,
        help=(
            "Smallest positive finite value retained during conversion. "
            f"Default: {MIN_FLOAT16_SUBNORMAL}."
        ),
    )
    parser.add_argument(
        "--max-finite-val",
        type=float,
        default=MAX_FLOAT16_FINITE,
        help=(
            "Largest finite magnitude retained during conversion. "
            f"Default: {MAX_FLOAT16_FINITE}."
        ),
    )
    parser.add_argument(
        "--runtime-provider",
        help=(
            "Optionally create an ONNX Runtime session before saving, for "
            "example CPUExecutionProvider or CUDAExecutionProvider."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output model, never the input model.",
    )
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    input_path = Path(args.input_model).expanduser().resolve()
    output_path = Path(args.output_model).expanduser().resolve()

    if not input_path.is_file():
        raise FileNotFoundError(f"Input model does not exist: {input_path}")
    if input_path.suffix.lower() != ".onnx":
        raise ValueError(f"Input model must have an .onnx suffix: {input_path}")
    if output_path.suffix.lower() != ".onnx":
        raise ValueError(f"Output model must have an .onnx suffix: {output_path}")
    if input_path == output_path:
        raise ValueError("Input and output model paths must be different.")
    if output_path.exists() and not output_path.is_file():
        raise ValueError(f"Output path exists and is not a file: {output_path}")
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output model already exists: {output_path}. Use --overwrite to replace it."
        )
    if args.min_positive_val < MIN_FLOAT16_SUBNORMAL:
        raise ValueError(
            "min-positive-val must be at least the smallest positive FP16 "
            f"subnormal ({MIN_FLOAT16_SUBNORMAL})."
        )
    if args.max_finite_val <= 0 or args.max_finite_val > MAX_FLOAT16_FINITE:
        raise ValueError(f"max-finite-val must be in (0, {MAX_FLOAT16_FINITE}].")
    if args.min_positive_val > args.max_finite_val:
        raise ValueError("min-positive-val must not exceed max-finite-val.")

    return input_path, output_path


def import_dependencies() -> tuple[Any, Any, Any]:
    try:
        import onnx
    except ImportError as error:
        raise ImportError(
            "onnx is required. Run with `uv run --with onnx --with onnxruntime`."
        ) from error

    try:
        import onnxruntime as ort
        from onnxruntime.transformers.float16 import convert_float_to_float16
    except ImportError as error:
        raise ImportError(
            "onnxruntime with its transformers tools is required. Run with "
            "`uv run --with onnx --with onnxruntime`."
        ) from error

    return onnx, ort, convert_float_to_float16


def tensor_type_name(value_info: Any, tensor_proto: Any) -> str:
    value_type = value_info.type
    if not value_type.HasField("tensor_type"):
        return "non-tensor"
    elem_type = value_type.tensor_type.elem_type
    if elem_type == tensor_proto.UNDEFINED:
        return "UNDEFINED"
    return tensor_proto.DataType.Name(elem_type)


def initializer_type_counts(model: Any, tensor_proto: Any) -> Dict[str, int]:
    counts = Counter(
        tensor_proto.DataType.Name(initializer.data_type)
        for initializer in model.graph.initializer
    )
    return dict(sorted(counts.items()))


def print_model_summary(label: str, model: Any, onnx: Any) -> None:
    tensor_proto = onnx.TensorProto
    opsets = [
        (opset.domain or "ai.onnx", opset.version)
        for opset in model.opset_import
    ]
    print(f"[info] {label} model: IR={model.ir_version}, opsets={opsets}")
    print(
        f"[info] {label} initializer dtypes: "
        f"{initializer_type_counts(model, tensor_proto)}"
    )
    print(f"[info] {label} graph inputs:")
    for value in model.graph.input:
        print(f"  {value.name}: {tensor_type_name(value, tensor_proto)}")
    print(f"[info] {label} graph outputs:")
    for value in model.graph.output:
        print(f"  {value.name}: {tensor_type_name(value, tensor_proto)}")


def iter_tensor_elem_types(values: Iterable[Any]) -> Iterable[int]:
    for value in values:
        value_type = value.type
        if value_type.HasField("tensor_type"):
            elem_type = value_type.tensor_type.elem_type
            if elem_type:
                yield elem_type


def check_source_has_fp32(model: Any, onnx: Any) -> None:
    tensor_proto = onnx.TensorProto
    value_types = set(
        iter_tensor_elem_types(
            list(model.graph.input)
            + list(model.graph.output)
            + list(model.graph.value_info)
        )
    )
    initializer_types = {
        initializer.data_type for initializer in model.graph.initializer
    }
    if tensor_proto.FLOAT not in value_types | initializer_types:
        raise ValueError(
            "The source model contains no FP32 tensor metadata or initializers; "
            "it does not appear to be an FP32 conversion source."
        )
    if tensor_proto.FLOAT16 in value_types | initializer_types:
        print(
            "[warning] Source model already contains FP16 tensors; conversion "
            "will produce a mixed-source result.",
            file=sys.stderr,
        )


def validate_runtime_model(model_path: Path, provider: str, ort: Any) -> None:
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise RuntimeError(
            f"Requested provider {provider!r} is unavailable; available providers: "
            f"{available_providers}"
        )
    session = ort.InferenceSession(
        str(model_path),
        providers=[provider],
    )
    print(
        f"[info] ONNX Runtime session check passed with {provider}: "
        f"{len(session.get_inputs())} inputs, {len(session.get_outputs())} outputs."
    )


def save_validated_model(
    model: Any,
    output_path: Path,
    overwrite: bool,
    runtime_provider: Optional[str],
    onnx: Any,
    ort: Any,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".onnx",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)

    try:
        onnx.save_model(model, str(temporary_path))
        onnx.checker.check_model(str(temporary_path), full_check=True)
        print("[info] Serialized FP16 model passed ONNX full_check.")

        if runtime_provider:
            validate_runtime_model(temporary_path, runtime_provider, ort)

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output model appeared during conversion: {output_path}. "
                "Use --overwrite to replace it."
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def convert_model(args: argparse.Namespace) -> Path:
    input_path, output_path = resolve_paths(args)
    onnx, ort, convert_float_to_float16 = import_dependencies()

    print(f"[info] onnx={onnx.__version__}, onnxruntime={ort.__version__}")
    print(f"[info] Loading source model: {input_path}")
    source_model = onnx.load(str(input_path))
    onnx.checker.check_model(source_model, full_check=True)
    print("[info] Source FP32 model passed ONNX full_check.")
    check_source_has_fp32(source_model, onnx)
    print_model_summary("source", source_model, onnx)

    converted_model = convert_float_to_float16(
        source_model,
        min_positive_val=args.min_positive_val,
        max_finite_val=args.max_finite_val,
        keep_io_types=args.keep_io_types,
        disable_shape_infer=args.disable_shape_infer,
    )
    onnx.checker.check_model(converted_model, full_check=True)
    print("[info] In-memory FP16 model passed ONNX full_check.")
    print_model_summary("converted", converted_model, onnx)

    save_validated_model(
        model=converted_model,
        output_path=output_path,
        overwrite=args.overwrite,
        runtime_provider=args.runtime_provider,
        onnx=onnx,
        ort=ort,
    )
    print(f"[success] Wrote validated FP16 model: {output_path}")
    return output_path


def main() -> None:
    args = parse_args()
    try:
        convert_model(args)
    except (FileNotFoundError, FileExistsError, ImportError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
