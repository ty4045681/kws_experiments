from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List, Tuple

CPU_PROVIDER = "CPUExecutionProvider"
CANN_PROVIDER = "CANNExecutionProvider"
SUPPORTED_PROVIDERS = (CPU_PROVIDER, CANN_PROVIDER)
CANN_PRECISION_MODES = (
    "force_fp32",
    "cube_fp16in_fp32out",
    "force_fp16",
    "allow_fp32_to_fp16",
    "must_keep_origin_dtype",
    "allow_mix_precision",
    "allow_mix_precision_fp16",
)

_CANN_ARGUMENTS = (
    ("cann_device_id", "--cann-device-id"),
    ("cann_npu_mem_limit", "--cann-npu-mem-limit"),
    ("cann_arena_extend_strategy", "--cann-arena-extend-strategy"),
    ("cann_enable_graph", "--cann-enable-graph/--no-cann-enable-graph"),
    ("cann_enable_subgraph", "--cann-enable-subgraph/--no-cann-enable-subgraph"),
    ("cann_precision_mode", "--cann-precision-mode"),
    ("cann_op_select_impl_mode", "--cann-op-select-impl-mode"),
    ("cann_optypelist_for_implmode", "--cann-optypelist-for-implmode"),
    ("cann_dump_graphs", "--cann-dump-graphs"),
    ("cann_dump_om_model", "--cann-dump-om-model"),
    ("cann_disable_cpu_fallback", "--cann-disable-cpu-fallback"),
)


def add_provider_arguments(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("ONNX Runtime execution provider")
    group.add_argument(
        "--provider",
        choices=SUPPORTED_PROVIDERS,
        default=CPU_PROVIDER,
        help=f"Execution provider. Default: {CPU_PROVIDER}.",
    )
    group.add_argument(
        "--cann-device-id",
        type=int,
        help="CANN device ID. Default: 0.",
    )
    group.add_argument(
        "--cann-npu-mem-limit",
        type=int,
        help="CANN device memory arena limit in bytes. Default: provider limit.",
    )
    group.add_argument(
        "--cann-arena-extend-strategy",
        choices=("kNextPowerOfTwo", "kSameAsRequested"),
        help="CANN memory arena growth strategy. Default: kNextPowerOfTwo.",
    )
    graph_group = group.add_mutually_exclusive_group()
    graph_group.add_argument(
        "--cann-enable-graph",
        dest="cann_enable_graph",
        action="store_true",
        default=None,
        help="Enable CANN graph execution. Enabled by default for CANN.",
    )
    graph_group.add_argument(
        "--no-cann-enable-graph",
        dest="cann_enable_graph",
        action="store_false",
        help="Disable CANN graph execution.",
    )
    subgraph_group = group.add_mutually_exclusive_group()
    subgraph_group.add_argument(
        "--cann-enable-subgraph",
        dest="cann_enable_subgraph",
        action="store_true",
        default=None,
        help="Allow automatic CANN/CPU subgraph partitioning. Disabled by default.",
    )
    subgraph_group.add_argument(
        "--no-cann-enable-subgraph",
        dest="cann_enable_subgraph",
        action="store_false",
        help="Disable automatic CANN subgraph partitioning.",
    )
    group.add_argument(
        "--cann-precision-mode",
        choices=CANN_PRECISION_MODES,
        help="CANN operator precision mode. Default: force_fp16.",
    )
    group.add_argument(
        "--cann-op-select-impl-mode",
        choices=("high_precision", "high_performance"),
        help="CANN operator implementation mode. Default: high_performance.",
    )
    group.add_argument(
        "--cann-optypelist-for-implmode",
        help="Comma-separated operator types governed by --cann-op-select-impl-mode.",
    )
    group.add_argument(
        "--cann-dump-graphs",
        action="store_true",
        default=None,
        help="Dump CANN-partitioned ONNX graphs.",
    )
    group.add_argument(
        "--cann-dump-om-model",
        action="store_true",
        default=None,
        help="Dump the compiled CANN OM model.",
    )
    group.add_argument(
        "--cann-disable-cpu-fallback",
        action="store_true",
        default=None,
        help="Fail session creation unless CANN supports the entire graph.",
    )


def validate_provider_arguments(args: argparse.Namespace) -> None:
    if args.provider == CPU_PROVIDER:
        specified = [
            option
            for destination, option in _CANN_ARGUMENTS
            if getattr(args, destination, None) is not None
        ]
        if specified:
            raise ValueError(
                "CANN-only options require --provider CANNExecutionProvider: "
                + ", ".join(specified)
            )
        return
    if args.cann_device_id is not None and args.cann_device_id < 0:
        raise ValueError("cann-device-id must be non-negative.")
    if args.cann_npu_mem_limit is not None and args.cann_npu_mem_limit < 1:
        raise ValueError("cann-npu-mem-limit must be positive.")
    if args.cann_optypelist_for_implmode == "":
        raise ValueError("cann-optypelist-for-implmode must not be empty.")


def _flag(value: bool) -> str:
    return "1" if value else "0"


def resolve_cann_options(args: argparse.Namespace) -> Dict[str, str]:
    options = {
        "device_id": str(args.cann_device_id if args.cann_device_id is not None else 0),
        "arena_extend_strategy": args.cann_arena_extend_strategy or "kNextPowerOfTwo",
        "enable_cann_graph": _flag(
            args.cann_enable_graph if args.cann_enable_graph is not None else True
        ),
        "enable_cann_subgraph": _flag(
            args.cann_enable_subgraph
            if args.cann_enable_subgraph is not None
            else False
        ),
        "dump_graphs": _flag(bool(args.cann_dump_graphs)),
        "dump_om_model": _flag(bool(args.cann_dump_om_model)),
        "precision_mode": args.cann_precision_mode or "force_fp16",
        "op_select_impl_mode": args.cann_op_select_impl_mode
        or "high_performance",
    }
    if args.cann_npu_mem_limit is not None:
        options["npu_mem_limit"] = str(args.cann_npu_mem_limit)
    if args.cann_optypelist_for_implmode is not None:
        options["optypelist_for_implmode"] = args.cann_optypelist_for_implmode
    return options


def create_inference_session(
    ort: Any,
    model_path: str,
    session_options: Any,
    args: argparse.Namespace,
) -> Tuple[Any, Dict[str, Any]]:
    validate_provider_arguments(args)
    available_providers = list(ort.get_available_providers())
    if args.provider not in available_providers:
        suffix = ""
        if args.provider == CANN_PROVIDER:
            suffix = (
                " Install an onnxruntime-cann wheel built for the server's CANN "
                "8.5.1 environment, or build ONNX Runtime with --use_cann."
            )
        raise RuntimeError(
            f"Requested provider {args.provider!r} is unavailable; "
            f"available providers: {available_providers}.{suffix}"
        )

    requested_options: Dict[str, str] = {}
    cpu_fallback_enabled = False
    if args.provider == CANN_PROVIDER:
        requested_options = resolve_cann_options(args)
        provider_specs: List[Any] = [(CANN_PROVIDER, requested_options)]
        if args.cann_disable_cpu_fallback:
            session_options.add_session_config_entry(
                "session.disable_cpu_ep_fallback", "1"
            )
        else:
            provider_specs.append(CPU_PROVIDER)
            cpu_fallback_enabled = True
    else:
        provider_specs = [CPU_PROVIDER]

    session = ort.InferenceSession(
        str(model_path),
        sess_options=session_options,
        providers=provider_specs,
    )
    metadata = {
        "onnxruntime_version": ort.__version__,
        "available_providers": available_providers,
        "requested_provider": args.provider,
        "requested_provider_options": requested_options,
        "session_providers": list(session.get_providers()),
        "session_provider_options": session.get_provider_options(),
        "cpu_fallback_enabled": cpu_fallback_enabled,
    }
    return session, metadata


def print_provider_metadata(metadata: Dict[str, Any]) -> None:
    print(f"[info] onnxruntime={metadata['onnxruntime_version']}")
    print(f"[info] Available providers: {metadata['available_providers']}")
    print(f"[info] Requested provider: {metadata['requested_provider']}")
    if metadata["requested_provider_options"]:
        options = json.dumps(
            metadata["requested_provider_options"],
            sort_keys=True,
            ensure_ascii=False,
        )
        print(f"[info] Requested provider options: {options}")
    print(f"[info] Session providers: {metadata['session_providers']}")
    if metadata["requested_provider"] == CANN_PROVIDER:
        state = "enabled" if metadata["cpu_fallback_enabled"] else "disabled"
        print(f"[info] CPU EP fallback: {state}")
        print(
            "[warning] Session providers show registration order, not per-node "
            "placement; use profiling or graph dumps to verify CANN execution."
        )
