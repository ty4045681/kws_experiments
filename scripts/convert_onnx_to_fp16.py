#!/usr/bin/env python3
r"""Convert a valid FP32 ONNX model to FP16 and optionally search for speed.

The converter uses ``onnxruntime.transformers.float16`` so existing
``Cast(to=FLOAT)`` nodes are updated consistently with FP16 value metadata.
Cast nodes inserted around FP32-only operators are recursively and stably
topologically sorted before validation. Both the source and converted models
are checked with ONNX full type/shape inference. The converted model is written
through a temporary file and moved to the requested destination only after all
enabled checks pass.

Example:
    uv run --with onnx==1.22.0 --with onnxruntime==1.28.0 \
      python scripts/convert_onnx_to_fp16.py \
      --input-model path/to/encoder.onnx \
      --output-model path/to/encoder.fp16.onnx \
      --runtime-provider CPUExecutionProvider

Use ``--keep-io-types`` when the deployment interface must retain FP32 graph
inputs and outputs while internal floating-point computation uses FP16.

To search mixed-precision variants, list operators or named nodes that may stay
in FP32 and provide a benchmark command. Every subset is converted and scored;
the fastest valid candidate is published to ``--output-model``::

    uv run --with onnx==1.22.0 --with onnxruntime==1.28.0 \
      python scripts/convert_onnx_to_fp16.py \
      --input-model path/to/encoder.onnx \
      --output-model path/to/encoder.fastest.onnx \
      --search-fp32-op Log --search-fp32-op Exp \
      --search-repeats 3 \
      --benchmark-command \
        'python scripts/bench_zipformer_encoder_onnx.py --encoder {model}'

The default metric parser reads ``p50: NUMBER`` from benchmark stdout. Command
tokens may contain ``{model}``, ``{candidate}``, ``{run}``, ``{run_dir}``,
``{work_dir}``, and ``{source}``. Commands run without a shell.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Pattern,
    Sequence,
    Tuple,
)


MIN_FLOAT16_SUBNORMAL = 5.96e-08
MAX_FLOAT16_FINITE = 65504.0
DEFAULT_BENCHMARK_METRIC_REGEX = (
    r"(?mi)^\s*p50\s*:\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
)


@dataclass
class TopologySortStats:
    """Summary of recursive graph topology normalization."""

    graphs_checked: int = 0
    graphs_reordered: int = 0
    nodes_repositioned: int = 0


@dataclass(frozen=True)
class PrecisionCandidate:
    """One mixed-precision configuration in an exhaustive search."""

    name: str
    fp32_ops: Tuple[str, ...]
    fp32_nodes: Tuple[str, ...]
    added_fp32_ops: Tuple[str, ...]
    added_fp32_nodes: Tuple[str, ...]


@dataclass
class CandidateBenchmark:
    """Benchmark outcome for one converted candidate."""

    candidate: PrecisionCandidate
    model_path: Path
    samples: List[float]
    score: Optional[float] = None
    error: Optional[str] = None


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

    precision_group = parser.add_argument_group("mixed-precision conversion")
    precision_group.add_argument(
        "--fp32-op",
        action="append",
        default=[],
        metavar="OP_TYPE",
        help=(
            "Keep every node of this ONNX operator type in FP32. Repeatable; "
            "the values are added to ONNX Runtime's safe default block list."
        ),
    )
    precision_group.add_argument(
        "--fp32-node",
        action="append",
        default=[],
        metavar="NODE_NAME",
        help="Keep this exactly named ONNX node in FP32. Repeatable.",
    )
    precision_group.add_argument(
        "--no-default-fp32-op-block-list",
        action="store_true",
        help=(
            "Do not include ONNX Runtime's default FP32 operator block list. "
            "Advanced and potentially numerically unsafe."
        ),
    )
    precision_group.add_argument(
        "--force-fp16-initializers",
        action="store_true",
        help=(
            "Force all FP32 initializers to FP16, including initializers used "
            "only by FP32-blocked nodes. This can add Cast nodes."
        ),
    )

    search_group = parser.add_argument_group("exhaustive performance search")
    search_group.add_argument(
        "--search-fp32-op",
        action="append",
        default=[],
        metavar="OP_TYPE",
        help=(
            "Search both FP16 and FP32-blocked variants of this operator type. "
            "Repeatable."
        ),
    )
    search_group.add_argument(
        "--search-fp32-node",
        action="append",
        default=[],
        metavar="NODE_NAME",
        help=(
            "Search both FP16 and FP32-blocked variants of this named node. Repeatable."
        ),
    )
    search_group.add_argument(
        "--benchmark-command",
        help=(
            "Shell-like command template used to score every candidate. It "
            "must contain {model}; the command is tokenized but not run in a shell."
        ),
    )
    search_group.add_argument(
        "--benchmark-metric-regex",
        default=DEFAULT_BENCHMARK_METRIC_REGEX,
        help=(
            "Regex whose first capture group is a numeric metric in benchmark "
            "stdout. Default extracts lines such as 'p50: 12.34'."
        ),
    )
    search_group.add_argument(
        "--benchmark-metric-reducer",
        choices=("mean", "median", "min", "max", "first", "last"),
        default="mean",
        help=(
            "Reduce multiple metric matches from one command. Default: mean "
            "(useful for multi-step fixture benchmarks)."
        ),
    )
    search_group.add_argument(
        "--benchmark-goal",
        choices=("min", "max"),
        default="min",
        help="Whether a lower or higher metric is better. Default: min.",
    )
    search_group.add_argument(
        "--benchmark-timeout",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="Timeout for each benchmark process. Default: 600 seconds.",
    )
    search_group.add_argument(
        "--benchmark-cwd",
        help="Working directory for benchmark commands. Default: current directory.",
    )
    search_group.add_argument(
        "--show-benchmark-output",
        action="store_true",
        help="Echo captured benchmark stdout and stderr for every run.",
    )
    search_group.add_argument(
        "--search-repeats",
        type=int,
        default=3,
        help="Benchmark process runs per candidate; scores use the median. Default: 3.",
    )
    search_group.add_argument(
        "--search-max-candidates",
        type=int,
        default=64,
        help="Refuse a larger exhaustive search. Default: 64.",
    )
    search_group.add_argument(
        "--search-work-dir",
        help=(
            "Parent directory for a retained search run. Without this option, "
            "candidate models and benchmark artifacts are temporary."
        ),
    )
    search_group.add_argument(
        "--search-report",
        help=(
            "Search report JSON path. Default: <output-model>.search.json when "
            "--benchmark-command is used."
        ),
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
            f"Output model already exists: {output_path}. "
            "Use --overwrite to replace it."
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

    for option_name in ("fp32_op", "fp32_node", "search_fp32_op", "search_fp32_node"):
        if any(not value for value in getattr(args, option_name)):
            raise ValueError(f"--{option_name.replace('_', '-')} must not be empty.")

    search_only_options_used = bool(
        args.search_fp32_op
        or args.search_fp32_node
        or args.search_work_dir
        or args.search_report
    )
    if search_only_options_used and not args.benchmark_command:
        raise ValueError(
            "Search options require --benchmark-command so candidates can be scored."
        )
    if args.benchmark_command:
        if "{model}" not in args.benchmark_command:
            raise ValueError(
                "--benchmark-command must contain the {model} placeholder."
            )
        if args.search_repeats < 1:
            raise ValueError("--search-repeats must be at least 1.")
        if args.search_max_candidates < 1:
            raise ValueError("--search-max-candidates must be at least 1.")
        if not math.isfinite(args.benchmark_timeout) or args.benchmark_timeout <= 0:
            raise ValueError("--benchmark-timeout must be a positive finite number.")
        try:
            metric_pattern = re.compile(args.benchmark_metric_regex)
        except re.error as error:
            raise ValueError(f"Invalid --benchmark-metric-regex: {error}") from error
        if metric_pattern.groups < 1:
            raise ValueError(
                "--benchmark-metric-regex must contain at least one capture group."
            )
        try:
            benchmark_tokens = shlex.split(args.benchmark_command)
        except ValueError as error:
            raise ValueError(f"Could not parse --benchmark-command: {error}") from error
        if not benchmark_tokens:
            raise ValueError("--benchmark-command must not be empty.")

    if args.benchmark_cwd:
        benchmark_cwd = Path(args.benchmark_cwd).expanduser().resolve()
        if not benchmark_cwd.is_dir():
            raise ValueError(
                f"Benchmark working directory does not exist: {benchmark_cwd}"
            )
    if args.search_work_dir:
        search_parent = Path(args.search_work_dir).expanduser().resolve()
        if search_parent.exists() and not search_parent.is_dir():
            raise ValueError(
                f"Search work directory exists and is not a directory: {search_parent}"
            )
    if args.search_report:
        report_path = Path(args.search_report).expanduser().resolve()
        if report_path in (input_path, output_path):
            raise ValueError("Search report path must differ from model paths.")
        if report_path.exists() and not args.overwrite:
            raise FileExistsError(
                f"Search report already exists: {report_path}. "
                "Use --overwrite to replace it."
            )

    return input_path, output_path


def import_dependencies() -> tuple[Any, Any, Any, Sequence[str]]:
    try:
        import onnx
    except ImportError as error:
        raise ImportError(
            "onnx is required. Run with `uv run --with onnx --with onnxruntime`."
        ) from error

    try:
        import onnxruntime as ort
        from onnxruntime.transformers.float16 import (
            DEFAULT_OP_BLOCK_LIST,
            convert_float_to_float16,
        )
    except ImportError as error:
        raise ImportError(
            "onnxruntime with its transformers tools is required. Run with "
            "`uv run --with onnx --with onnxruntime`."
        ) from error

    return onnx, ort, convert_float_to_float16, DEFAULT_OP_BLOCK_LIST


def node_label(node: Any, index: int) -> str:
    """Return a useful node label for diagnostics."""

    return node.name or f"{node.op_type}[{index}]"


def stable_topological_sort_graph(
    graph: Any,
    attribute_proto: Any,
    stats: TopologySortStats,
    graph_path: str,
) -> None:
    """Recursively put one GraphProto's nodes in stable topological order.

    Dependencies are created only for values produced by nodes in the same
    graph. Graph inputs, initializers, and values captured from an outer graph
    therefore remain valid roots. Among nodes that are ready at the same time,
    the original node index is used to preserve deterministic ordering.
    """

    stats.graphs_checked += 1
    nodes = list(graph.node)

    for node_index, node in enumerate(nodes):
        parent_label = node_label(node, node_index)
        for attribute in node.attribute:
            if attribute.type == attribute_proto.GRAPH:
                child_name = attribute.g.name or attribute.name or "<graph>"
                stable_topological_sort_graph(
                    attribute.g,
                    attribute_proto,
                    stats,
                    f"{graph_path}/{parent_label}:{child_name}",
                )
            elif attribute.type == attribute_proto.GRAPHS:
                for child_index, child_graph in enumerate(attribute.graphs):
                    child_name = child_graph.name or f"{attribute.name}[{child_index}]"
                    stable_topological_sort_graph(
                        child_graph,
                        attribute_proto,
                        stats,
                        f"{graph_path}/{parent_label}:{child_name}",
                    )

    if len(nodes) < 2:
        return

    producer_by_value: Dict[str, int] = {}
    for producer_index, node in enumerate(nodes):
        for output_name in node.output:
            if not output_name:
                continue
            previous_index = producer_by_value.get(output_name)
            if previous_index is not None:
                raise RuntimeError(
                    f"Cannot topologically sort {graph_path}: value "
                    f"{output_name!r} is produced by both "
                    f"{node_label(nodes[previous_index], previous_index)!r} and "
                    f"{node_label(node, producer_index)!r}."
                )
            producer_by_value[output_name] = producer_index

    indegrees = [0] * len(nodes)
    consumers_by_producer = [[] for _ in nodes]
    for consumer_index, node in enumerate(nodes):
        dependencies = {
            producer_by_value[input_name]
            for input_name in node.input
            if input_name in producer_by_value
        }
        indegrees[consumer_index] = len(dependencies)
        for producer_index in dependencies:
            consumers_by_producer[producer_index].append(consumer_index)

    ready = [index for index, indegree in enumerate(indegrees) if indegree == 0]
    heapq.heapify(ready)
    sorted_indices = []
    while ready:
        producer_index = heapq.heappop(ready)
        sorted_indices.append(producer_index)
        for consumer_index in consumers_by_producer[producer_index]:
            indegrees[consumer_index] -= 1
            if indegrees[consumer_index] == 0:
                heapq.heappush(ready, consumer_index)

    if len(sorted_indices) != len(nodes):
        blocked_nodes = [
            node_label(nodes[index], index)
            for index, indegree in enumerate(indegrees)
            if indegree > 0
        ]
        preview = ", ".join(repr(name) for name in blocked_nodes[:10])
        suffix = " ..." if len(blocked_nodes) > 10 else ""
        raise RuntimeError(
            f"Cannot topologically sort {graph_path}: detected a dependency "
            f"cycle involving {preview}{suffix}."
        )

    original_indices = list(range(len(nodes)))
    if sorted_indices == original_indices:
        return

    reordered_nodes = []
    for original_index in sorted_indices:
        node_copy = type(nodes[original_index])()
        node_copy.CopyFrom(nodes[original_index])
        reordered_nodes.append(node_copy)

    del graph.node[:]
    graph.node.extend(reordered_nodes)
    stats.graphs_reordered += 1
    stats.nodes_repositioned += sum(
        original_index != new_index
        for new_index, original_index in enumerate(sorted_indices)
    )


def stable_topological_sort_model(model: Any, onnx: Any) -> TopologySortStats:
    """Recursively normalize node order in the model's main graph."""

    stats = TopologySortStats()
    graph_name = model.graph.name or "<main>"
    stable_topological_sort_graph(
        model.graph,
        onnx.AttributeProto,
        stats,
        f"graph:{graph_name}",
    )
    return stats


def validate_onnx_model(model_or_path: Any, label: str, onnx: Any) -> None:
    """Run ONNX full validation and convert checker errors to concise output."""

    try:
        onnx.checker.check_model(model_or_path, full_check=True)
    except Exception as error:
        raise RuntimeError(f"{label} failed ONNX full_check: {error}") from error
    print(f"[info] {label} passed ONNX full_check.")


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
        (opset.domain or "ai.onnx", opset.version) for opset in model.opset_import
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


def unique_values(values: Iterable[str]) -> Tuple[str, ...]:
    """Deduplicate CLI values without changing their order."""

    return tuple(dict.fromkeys(values))


def iter_model_nodes(graph: Any, attribute_proto: Any) -> Iterator[Any]:
    """Yield nodes from the main graph and every nested GraphProto."""

    for node in graph.node:
        yield node
        for attribute in node.attribute:
            if attribute.type == attribute_proto.GRAPH:
                yield from iter_model_nodes(attribute.g, attribute_proto)
            elif attribute.type == attribute_proto.GRAPHS:
                for child_graph in attribute.graphs:
                    yield from iter_model_nodes(child_graph, attribute_proto)


def prepare_precision_candidates(
    source_model: Any,
    args: argparse.Namespace,
    default_op_block_list: Sequence[str],
    onnx: Any,
) -> List[PrecisionCandidate]:
    """Validate requested targets and create every search-space subset."""

    fixed_ops = unique_values(args.fp32_op)
    fixed_nodes = unique_values(args.fp32_node)
    search_ops = unique_values(args.search_fp32_op)
    search_nodes = unique_values(args.search_fp32_node)

    all_nodes = list(iter_model_nodes(source_model.graph, onnx.AttributeProto))
    source_ops = {node.op_type for node in all_nodes}
    source_node_counts = Counter(node.name for node in all_nodes if node.name)

    missing_fixed_ops = [name for name in fixed_ops if name not in source_ops]
    if missing_fixed_ops:
        print(
            "[warning] Fixed FP32 operator types are absent from the source model: "
            f"{missing_fixed_ops}",
            file=sys.stderr,
        )
    missing_search_ops = [name for name in search_ops if name not in source_ops]
    if missing_search_ops:
        raise ValueError(
            "Search FP32 operator types are absent from the source model: "
            f"{missing_search_ops}"
        )

    requested_nodes = fixed_nodes + search_nodes
    missing_nodes = [name for name in requested_nodes if name not in source_node_counts]
    if missing_nodes:
        raise ValueError(
            "Requested FP32 node names are absent from the source model: "
            f"{missing_nodes}"
        )
    duplicate_node_names = [
        name for name in requested_nodes if source_node_counts[name] > 1
    ]
    if duplicate_node_names:
        print(
            "[warning] These node names are not unique; each setting affects all "
            f"matching nodes: {unique_values(duplicate_node_names)}",
            file=sys.stderr,
        )

    if args.no_default_fp32_op_block_list:
        base_ops = fixed_ops
    else:
        base_ops = unique_values(tuple(default_op_block_list) + fixed_ops)
    base_nodes = fixed_nodes

    redundant_search_ops = [name for name in search_ops if name in base_ops]
    redundant_search_nodes = [name for name in search_nodes if name in base_nodes]
    if redundant_search_ops or redundant_search_nodes:
        details = []
        if redundant_search_ops:
            details.append(f"operators={redundant_search_ops}")
        if redundant_search_nodes:
            details.append(f"nodes={redundant_search_nodes}")
        raise ValueError(
            "Search dimensions are already always blocked as FP32: "
            + ", ".join(details)
        )

    dimensions = [("op", name) for name in search_ops]
    dimensions.extend(("node", name) for name in search_nodes)
    candidate_count = 2 ** len(dimensions)
    if candidate_count > args.search_max_candidates:
        raise ValueError(
            f"Search has {candidate_count} candidates ({len(dimensions)} binary "
            "dimensions), exceeding --search-max-candidates="
            f"{args.search_max_candidates}."
        )

    width = max(3, len(str(candidate_count - 1)))
    candidates = []
    for candidate_index, enabled_flags in enumerate(
        itertools.product((False, True), repeat=len(dimensions))
    ):
        added_ops = tuple(
            name
            for enabled, (kind, name) in zip(enabled_flags, dimensions)
            if enabled and kind == "op"
        )
        added_nodes = tuple(
            name
            for enabled, (kind, name) in zip(enabled_flags, dimensions)
            if enabled and kind == "node"
        )
        candidates.append(
            PrecisionCandidate(
                name=f"candidate-{candidate_index:0{width}d}",
                fp32_ops=unique_values(base_ops + added_ops),
                fp32_nodes=unique_values(base_nodes + added_nodes),
                added_fp32_ops=added_ops,
                added_fp32_nodes=added_nodes,
            )
        )
    return candidates


def validate_runtime_model(
    model_path: Path,
    provider: str,
    ort: Any,
    announce: bool = True,
) -> None:
    available_providers = ort.get_available_providers()
    if provider not in available_providers:
        raise RuntimeError(
            f"Requested provider {provider!r} is unavailable; available providers: "
            f"{available_providers}"
        )
    try:
        session = ort.InferenceSession(
            str(model_path),
            providers=[provider],
        )
    except Exception as error:
        raise RuntimeError(
            f"ONNX Runtime session check failed with {provider}: {error}"
        ) from error
    if announce:
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
    announce: bool = True,
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
        if announce:
            validate_onnx_model(
                str(temporary_path),
                "Serialized FP16 model",
                onnx,
            )
        else:
            try:
                onnx.checker.check_model(str(temporary_path), full_check=True)
            except Exception as error:
                raise RuntimeError(
                    f"Serialized FP16 model failed ONNX full_check: {error}"
                ) from error

        if runtime_provider:
            validate_runtime_model(
                temporary_path,
                runtime_provider,
                ort,
                announce=announce,
            )

        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output model appeared during conversion: {output_path}. "
                "Use --overwrite to replace it."
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def load_source_model(input_path: Path, onnx: Any) -> Any:
    """Load one fresh model copy for conversion."""

    try:
        return onnx.load(str(input_path))
    except Exception as error:
        raise RuntimeError(f"Could not load source ONNX model: {error}") from error


def convert_source_model(
    source_model: Any,
    candidate: PrecisionCandidate,
    args: argparse.Namespace,
    convert_float_to_float16: Any,
    onnx: Any,
    announce: bool,
) -> tuple[Any, TopologySortStats]:
    """Convert and normalize one candidate from an unmodified source model."""

    conversion_options = {
        "min_positive_val": args.min_positive_val,
        "max_finite_val": args.max_finite_val,
        "keep_io_types": args.keep_io_types,
        "disable_shape_infer": args.disable_shape_infer,
        "op_block_list": list(candidate.fp32_ops),
        "node_block_list": list(candidate.fp32_nodes),
    }
    if args.force_fp16_initializers:
        conversion_options["force_fp16_initializers"] = True

    try:
        converted_model = convert_float_to_float16(
            source_model,
            **conversion_options,
        )
    except Exception as error:
        raise RuntimeError(f"FP16 conversion failed: {error}") from error

    topology_stats = stable_topological_sort_model(converted_model, onnx)
    if announce:
        print(
            "[info] Topological sort checked "
            f"{topology_stats.graphs_checked} graph(s), reordered "
            f"{topology_stats.graphs_reordered} graph(s), and repositioned "
            f"{topology_stats.nodes_repositioned} node(s)."
        )
        validate_onnx_model(converted_model, "In-memory FP16 model", onnx)
    else:
        try:
            onnx.checker.check_model(converted_model, full_check=True)
        except Exception as error:
            raise RuntimeError(
                f"In-memory FP16 model failed ONNX full_check: {error}"
            ) from error
    return converted_model, topology_stats


def reduce_metric(values: Sequence[float], reducer: str) -> float:
    """Reduce all metric matches emitted by one benchmark invocation."""

    if reducer == "mean":
        return statistics.mean(values)
    if reducer == "median":
        return statistics.median(values)
    if reducer == "min":
        return min(values)
    if reducer == "max":
        return max(values)
    if reducer == "first":
        return values[0]
    if reducer == "last":
        return values[-1]
    raise ValueError(f"Unsupported benchmark metric reducer: {reducer}")


def extract_benchmark_metric(
    stdout: str,
    pattern: Pattern[str],
    reducer: str,
) -> float:
    """Extract finite numeric values from stdout and reduce them to one score."""

    values = []
    for match in pattern.finditer(stdout):
        try:
            value = float(match.group(1))
        except (IndexError, ValueError) as error:
            raise RuntimeError(
                "The first benchmark regex capture group must be numeric."
            ) from error
        if not math.isfinite(value):
            raise RuntimeError(f"Benchmark metric must be finite, got {value!r}.")
        values.append(value)
    if not values:
        raise RuntimeError("Benchmark stdout did not match --benchmark-metric-regex.")
    return float(reduce_metric(values, reducer))


def output_tail(value: Any, limit: int = 4000) -> str:
    """Keep process diagnostics useful without making reports unbounded."""

    if not value:
        return ""
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    elif not isinstance(value, str):
        value = str(value)
    value = value.strip()
    if len(value) <= limit:
        return value
    return "..." + value[-limit:]


def expand_benchmark_command(
    command_tokens: Sequence[str],
    candidate: PrecisionCandidate,
    model_path: Path,
    run_index: int,
    run_dir: Path,
    work_dir: Path,
    input_path: Path,
) -> List[str]:
    """Expand known placeholders after tokenization so paths stay single argv items."""

    replacements = {
        "{model}": str(model_path),
        "{candidate}": candidate.name,
        "{run}": str(run_index + 1),
        "{run_dir}": str(run_dir),
        "{work_dir}": str(work_dir),
        "{source}": str(input_path),
    }
    expanded = []
    for original_token in command_tokens:
        token = original_token
        for placeholder, replacement in replacements.items():
            token = token.replace(placeholder, replacement)
        expanded.append(token)
    return expanded


def echo_benchmark_output(candidate_name: str, run_index: int, process: Any) -> None:
    """Print captured benchmark output only when explicitly requested."""

    if process.stdout:
        print(f"[benchmark:{candidate_name}:run-{run_index + 1}:stdout]")
        print(process.stdout.rstrip())
    if process.stderr:
        print(
            f"[benchmark:{candidate_name}:run-{run_index + 1}:stderr]",
            file=sys.stderr,
        )
        print(process.stderr.rstrip(), file=sys.stderr)


def run_benchmark_process(
    command: Sequence[str],
    args: argparse.Namespace,
    metric_pattern: Pattern[str],
) -> tuple[Optional[float], Optional[str], Any]:
    """Run one benchmark command and return its metric or a concise error."""

    benchmark_cwd = None
    if args.benchmark_cwd:
        benchmark_cwd = str(Path(args.benchmark_cwd).expanduser().resolve())
    try:
        process = subprocess.run(
            list(command),
            cwd=benchmark_cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=args.benchmark_timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as error:
        details = output_tail(error.stdout) or output_tail(error.stderr)
        suffix = f" Output: {details}" if details else ""
        return (
            None,
            f"benchmark timed out after {args.benchmark_timeout:g}s.{suffix}",
            None,
        )
    except OSError as error:
        return None, f"could not start benchmark command: {error}", None

    if process.returncode != 0:
        details = output_tail(process.stderr) or output_tail(process.stdout)
        suffix = f" Output: {details}" if details else ""
        return (
            None,
            f"benchmark exited with code {process.returncode}.{suffix}",
            process,
        )
    try:
        metric = extract_benchmark_metric(
            process.stdout,
            metric_pattern,
            args.benchmark_metric_reducer,
        )
    except RuntimeError as error:
        details = output_tail(process.stdout)
        suffix = f" Stdout: {details}" if details else ""
        return None, f"{error}{suffix}", process
    return metric, None, process


def benchmark_candidates(
    results: List[CandidateBenchmark],
    args: argparse.Namespace,
    input_path: Path,
    work_dir: Path,
) -> None:
    """Benchmark candidates in rotated round-robin order to reduce order bias."""

    metric_pattern = re.compile(args.benchmark_metric_regex)
    command_tokens = shlex.split(args.benchmark_command)
    runnable_results = [result for result in results if result.error is None]
    if not runnable_results:
        return

    for run_index in range(args.search_repeats):
        offset = run_index % len(runnable_results)
        run_order = runnable_results[offset:] + runnable_results[:offset]
        print(f"[search] Benchmark round {run_index + 1}/{args.search_repeats}")
        for result in run_order:
            if result.error is not None:
                continue
            run_dir = result.model_path.parent / f"benchmark-run-{run_index + 1:03d}"
            run_dir.mkdir(parents=True, exist_ok=False)
            command = expand_benchmark_command(
                command_tokens,
                result.candidate,
                result.model_path,
                run_index,
                run_dir,
                work_dir,
                input_path,
            )
            metric, error, process = run_benchmark_process(
                command,
                args,
                metric_pattern,
            )
            if args.show_benchmark_output and process is not None:
                echo_benchmark_output(result.candidate.name, run_index, process)
            if error:
                result.error = error
                print(
                    f"[warning] {result.candidate.name} failed: {error}",
                    file=sys.stderr,
                )
                continue
            assert metric is not None
            result.samples.append(metric)
            print(
                f"[search] {result.candidate.name} run {run_index + 1}: "
                f"metric={metric:.9g}"
            )

    for result in results:
        if result.error is None and len(result.samples) == args.search_repeats:
            result.score = float(statistics.median(result.samples))
        elif result.error is None:
            result.error = (
                f"completed only {len(result.samples)}/{args.search_repeats} repeats"
            )


@contextmanager
def create_search_workspace(
    args: argparse.Namespace,
    output_path: Path,
) -> Iterator[tuple[Path, bool]]:
    """Create a disposable or explicitly retained candidate workspace."""

    prefix = f"{output_path.stem}.search-"
    if args.search_work_dir:
        parent = Path(args.search_work_dir).expanduser().resolve()
        parent.mkdir(parents=True, exist_ok=True)
        workspace = Path(tempfile.mkdtemp(prefix=prefix, dir=parent))
        yield workspace, True
        return
    with tempfile.TemporaryDirectory(prefix=prefix) as temporary_name:
        yield Path(temporary_name), False


def publish_candidate_model(
    candidate_path: Path,
    output_path: Path,
    overwrite: bool,
    onnx: Any,
) -> None:
    """Atomically publish the exact candidate file that was benchmarked."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.",
        suffix=".onnx",
        dir=output_path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        shutil.copyfile(candidate_path, temporary_path)
        validate_onnx_model(
            str(temporary_path),
            "Selected serialized FP16 model",
            onnx,
        )
        if output_path.exists() and not overwrite:
            raise FileExistsError(
                f"Output model appeared during search: {output_path}. "
                "Use --overwrite to replace it."
            )
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def write_json_atomic(payload: Dict[str, Any], path: Path, overwrite: bool) -> None:
    """Write a JSON report without exposing a partially written destination."""

    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.stem}.",
        suffix=".json",
        dir=path.parent,
    )
    os.close(file_descriptor)
    temporary_path = Path(temporary_name)
    try:
        temporary_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        if path.exists() and not overwrite:
            raise FileExistsError(
                f"Search report appeared during search: {path}. "
                "Use --overwrite to replace it."
            )
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def search_report_payload(
    results: Sequence[CandidateBenchmark],
    winner: CandidateBenchmark,
    args: argparse.Namespace,
    input_path: Path,
    output_path: Path,
    work_dir: Path,
    retained: bool,
    onnx: Any,
    ort: Any,
) -> Dict[str, Any]:
    """Build a reproducible machine-readable search summary."""

    candidate_records = []
    for result in results:
        candidate_records.append(
            {
                "name": result.candidate.name,
                "status": "ok" if result.score is not None else "failed",
                "score": result.score,
                "samples": result.samples,
                "fp32_ops": list(result.candidate.fp32_ops),
                "fp32_nodes": list(result.candidate.fp32_nodes),
                "added_fp32_ops": list(result.candidate.added_fp32_ops),
                "added_fp32_nodes": list(result.candidate.added_fp32_nodes),
                "model": str(result.model_path) if retained else None,
                "error": result.error,
            }
        )
    return {
        "format_version": 1,
        "source_model": str(input_path),
        "output_model": str(output_path),
        "onnx_version": onnx.__version__,
        "onnxruntime_version": ort.__version__,
        "search_work_dir": str(work_dir) if retained else None,
        "conversion": {
            "keep_io_types": args.keep_io_types,
            "disable_shape_infer": args.disable_shape_infer,
            "min_positive_val": args.min_positive_val,
            "max_finite_val": args.max_finite_val,
            "force_fp16_initializers": args.force_fp16_initializers,
            "default_fp32_op_block_list_enabled": (
                not args.no_default_fp32_op_block_list
            ),
            "fixed_fp32_ops": list(unique_values(args.fp32_op)),
            "fixed_fp32_nodes": list(unique_values(args.fp32_node)),
            "search_fp32_ops": list(unique_values(args.search_fp32_op)),
            "search_fp32_nodes": list(unique_values(args.search_fp32_node)),
        },
        "benchmark": {
            "command_template": args.benchmark_command,
            "cwd": str(Path(args.benchmark_cwd).expanduser().resolve())
            if args.benchmark_cwd
            else str(Path.cwd()),
            "metric_regex": args.benchmark_metric_regex,
            "metric_reducer": args.benchmark_metric_reducer,
            "goal": args.benchmark_goal,
            "repeats": args.search_repeats,
            "timeout_seconds": args.benchmark_timeout,
            "aggregate_across_repeats": "median",
        },
        "winner": {
            "name": winner.candidate.name,
            "score": winner.score,
            "samples": winner.samples,
            "fp32_ops": list(winner.candidate.fp32_ops),
            "fp32_nodes": list(winner.candidate.fp32_nodes),
            "added_fp32_ops": list(winner.candidate.added_fp32_ops),
            "added_fp32_nodes": list(winner.candidate.added_fp32_nodes),
        },
        "candidates": candidate_records,
    }


def run_performance_search(
    args: argparse.Namespace,
    candidates: Sequence[PrecisionCandidate],
    input_path: Path,
    output_path: Path,
    onnx: Any,
    ort: Any,
    convert_float_to_float16: Any,
) -> Path:
    """Convert, benchmark, rank, and publish an exhaustive candidate set."""

    report_path = (
        Path(args.search_report).expanduser().resolve()
        if args.search_report
        else Path(f"{output_path}.search.json")
    )
    if report_path in (input_path, output_path):
        raise ValueError("Search report path must differ from model paths.")
    if report_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Search report already exists: {report_path}. "
            "Use --overwrite to replace it."
        )

    print(
        f"[search] Exhaustive search: {len(candidates)} candidate(s), "
        f"{args.search_repeats} benchmark repeat(s) each."
    )
    with create_search_workspace(args, output_path) as (work_dir, retained):
        workspace_kind = "retained" if retained else "temporary"
        print(f"[search] Workspace: {work_dir} ({workspace_kind})")
        results = []
        for candidate in candidates:
            candidate_dir = work_dir / candidate.name
            candidate_dir.mkdir(parents=True, exist_ok=False)
            candidate_path = candidate_dir / "model.onnx"
            result = CandidateBenchmark(candidate, candidate_path, [])
            results.append(result)
            print(
                f"[search] Converting {candidate.name}: "
                f"added_fp32_ops={list(candidate.added_fp32_ops)}, "
                f"added_fp32_nodes={list(candidate.added_fp32_nodes)}"
            )
            try:
                candidate_source = load_source_model(input_path, onnx)
                converted_model, topology_stats = convert_source_model(
                    candidate_source,
                    candidate,
                    args,
                    convert_float_to_float16,
                    onnx,
                    announce=False,
                )
                save_validated_model(
                    model=converted_model,
                    output_path=candidate_path,
                    overwrite=False,
                    runtime_provider=args.runtime_provider,
                    onnx=onnx,
                    ort=ort,
                    announce=False,
                )
                print(
                    f"[search] {candidate.name} ready; topological sort reordered "
                    f"{topology_stats.graphs_reordered} graph(s), repositioned "
                    f"{topology_stats.nodes_repositioned} node(s)."
                )
            except Exception as error:
                result.error = f"conversion or validation failed: {error}"
                print(
                    f"[warning] {candidate.name} failed: {result.error}",
                    file=sys.stderr,
                )

        benchmark_candidates(results, args, input_path, work_dir)
        successful_results = [result for result in results if result.score is not None]
        if not successful_results:
            failures = "; ".join(
                f"{result.candidate.name}: {result.error}" for result in results
            )
            raise RuntimeError(
                f"No search candidate completed successfully. {failures}"
            )

        def score_key(result: CandidateBenchmark) -> float:
            assert result.score is not None
            return result.score

        if args.benchmark_goal == "min":
            winner = min(successful_results, key=score_key)
        else:
            winner = max(successful_results, key=score_key)
        assert winner.score is not None

        print("[search] Ranking:")
        reverse = args.benchmark_goal == "max"
        for rank, result in enumerate(
            sorted(successful_results, key=score_key, reverse=reverse),
            start=1,
        ):
            print(
                f"  {rank:>2}. {result.candidate.name}: score={result.score:.9g}, "
                f"added_fp32_ops={list(result.candidate.added_fp32_ops)}, "
                f"added_fp32_nodes={list(result.candidate.added_fp32_nodes)}"
            )

        publish_candidate_model(
            winner.model_path,
            output_path,
            args.overwrite,
            onnx,
        )
        report = search_report_payload(
            results,
            winner,
            args,
            input_path,
            output_path,
            work_dir,
            retained,
            onnx,
            ort,
        )
        write_json_atomic(report, report_path, args.overwrite)
        print(
            f"[success] Selected {winner.candidate.name} with score "
            f"{winner.score:.9g}: {output_path}"
        )
        print(f"[success] Wrote search report: {report_path}")
        if retained:
            print(f"[success] Retained candidate artifacts: {work_dir}")
    return output_path


def convert_model(args: argparse.Namespace) -> Path:
    input_path, output_path = resolve_paths(args)
    (
        onnx,
        ort,
        convert_float_to_float16,
        default_op_block_list,
    ) = import_dependencies()

    print(f"[info] onnx={onnx.__version__}, onnxruntime={ort.__version__}")
    print(f"[info] Loading source model: {input_path}")
    source_model = load_source_model(input_path, onnx)
    validate_onnx_model(source_model, "Source FP32 model", onnx)
    check_source_has_fp32(source_model, onnx)
    print_model_summary("source", source_model, onnx)

    candidates = prepare_precision_candidates(
        source_model,
        args,
        default_op_block_list,
        onnx,
    )
    if args.benchmark_command:
        return run_performance_search(
            args,
            candidates,
            input_path,
            output_path,
            onnx,
            ort,
            convert_float_to_float16,
        )

    candidate = candidates[0]
    converted_model, _ = convert_source_model(
        source_model,
        candidate,
        args,
        convert_float_to_float16,
        onnx,
        announce=True,
    )
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
    except (
        FileNotFoundError,
        FileExistsError,
        ImportError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
