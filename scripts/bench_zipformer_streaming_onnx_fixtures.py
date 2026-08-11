#!/usr/bin/env python3
"""Benchmark a streaming Zipformer ONNX encoder with pre-generated fixtures.

Reads fixtures produced by generate_zipformer_streaming_fixtures.py, runs each
step with the ONNX Runtime Python API (warmup + timed loops per step), and
saves output tensors for later accuracy comparison against MindIR benchmark.

Example:
    python scripts/bench_zipformer_streaming_onnx_fixtures.py \
        --fixtures-dir fixtures/zipformer
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures-dir", required=True, help="Fixture directory containing manifest.json.")
    parser.add_argument("--model", help="Path to the ONNX encoder model. Default: the model recorded in the manifest.")
    parser.add_argument("--warmup", type=int, default=20, help="Warmup runs per step. Default: 20.")
    parser.add_argument("--loops", type=int, default=100, help="Timed runs per step. Default: 100.")
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
    parser.add_argument("--output-dir", help="Directory for saved output tensors. Default: <fixtures-dir>/onnx_outputs.")
    parser.add_argument("--overwrite", action="store_true", help="Allow writing into a non-empty output directory.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.warmup < 0 or args.loops < 1:
        raise ValueError("warmup must be non-negative and loops must be positive.")
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


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._") or "tensor"


def load_manifest(fixtures_dir: Path) -> Dict[str, Any]:
    manifest_path = fixtures_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if "onnx" not in manifest.get("backends", {}):
        raise ValueError(f"Manifest has no ONNX backend fixtures: {manifest_path}")
    return manifest


def load_step_feed(fixtures_dir: Path, fixture: Dict[str, Any]) -> Dict[str, np.ndarray]:
    feed: Dict[str, np.ndarray] = {}
    for entry in fixture["inputs"]:
        path = fixtures_dir / entry["file"]
        if not path.is_file():
            raise FileNotFoundError(f"Fixture input file does not exist: {path}")
        array = np.fromfile(path, dtype=np.dtype(entry["dtype_str"]))
        if array.size != entry["element_count"]:
            raise ValueError(f"Fixture input {path} has {array.size} elements; manifest expects {entry['element_count']}.")
        feed[entry["name"]] = array.reshape(entry["shape"])
    return feed


def check_feed(session: Any, feed: Dict[str, np.ndarray], step_name: str) -> None:
    model_names = [meta.name for meta in session.get_inputs()]
    if set(model_names) != set(feed):
        raise ValueError(f"{step_name}: fixture inputs {sorted(feed)} do not match model inputs {sorted(model_names)}.")


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


def bench_step(session: Any, output_names: List[str], feed: Dict[str, np.ndarray], args: argparse.Namespace) -> Tuple[List[float], List[np.ndarray]]:
    for _ in range(args.warmup):
        session.run(output_names, feed)
    latencies_ms = []
    for _ in range(args.loops):
        started = time.perf_counter()
        session.run(output_names, feed)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
    outputs = session.run(output_names, feed)
    return latencies_ms, outputs


def save_outputs(output_dir: Path, step_name: str, output_names: List[str], outputs: List[np.ndarray]) -> Dict[str, Any]:
    directory = output_dir / step_name
    directory.mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (name, value) in enumerate(zip(output_names, outputs)):
        array = np.ascontiguousarray(value)
        path = directory / f"output_{index:02d}_{safe_name(name)}.bin"
        array.tofile(path)
        entries.append({
            "index": index,
            "name": name,
            "dtype": str(array.dtype),
            "dtype_str": array.dtype.str,
            "shape": list(array.shape),
            "element_count": int(array.size),
            "byte_size": int(array.nbytes),
            "file": str(path.relative_to(output_dir)),
        })
    return {"name": step_name, "outputs": entries}


def print_step_summary(step_name: str, latencies_ms: List[float]) -> Dict[str, float]:
    summary = summarize(latencies_ms)
    print(f"\n[result] {step_name}: latency over {len(latencies_ms)} runs (ms)")
    for name in ("mean", "std", "min", "p50", "p90", "p95", "p99", "max"):
        print(f"  {name:>4}: {summary[name]:.3f}")
    return summary


def prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise ValueError(f"Output directory is not empty: {path}. Use --overwrite to write into it.")
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    args = parse_args()
    try:
        validate_args(args)
        fixtures_dir = Path(args.fixtures_dir).resolve()
        manifest = load_manifest(fixtures_dir)
        backend = manifest["backends"]["onnx"]
        model_path = Path(args.model) if args.model else Path(backend["model"])
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file does not exist: {model_path}")
        output_dir = Path(args.output_dir).resolve() if args.output_dir else fixtures_dir / "onnx_outputs"
        prepare_output_dir(output_dir, args.overwrite)
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
        session = ort.InferenceSession(str(model_path), sess_options=options, providers=["CPUExecutionProvider"])
        if args.disable_optimizer:
            print(f"[info] Disabled optimizers: {args.disable_optimizer}")
        output_names = [meta.name for meta in session.get_outputs()]
        print(f"[info] Model: {model_path}")
        print(f"[info] Fixtures: {fixtures_dir} ({len(backend['fixtures'])} steps)")
        print(f"[info] warmup={args.warmup}, loops={args.loops}, threads={args.threads}, inter_op_threads={args.inter_op_threads}")

        steps = []
        for fixture in backend["fixtures"]:
            step_name = fixture["name"]
            feed = load_step_feed(fixtures_dir, fixture)
            check_feed(session, feed, step_name)
            latencies_ms, outputs = bench_step(session, output_names, feed, args)
            summary = print_step_summary(step_name, latencies_ms)
            record = save_outputs(output_dir, step_name, output_names, outputs)
            record.update({"step": fixture["step"], "latency_ms": summary, "runs": len(latencies_ms)})
            steps.append(record)

        profile_path = finish_profiling(session, args.profile)
        results = {
            "format_version": 1,
            "generator": Path(__file__).name,
            "model": str(model_path.resolve()),
            "fixtures_dir": str(fixtures_dir),
            "configuration": {
                "warmup": args.warmup,
                "loops": args.loops,
                "threads": args.threads,
                "inter_op_threads": args.inter_op_threads,
                "disabled_optimizers": list(args.disable_optimizer),
                "profiling": args.profile,
                "profile_file": profile_path,
            },
            "steps": steps,
        }
        (output_dir / "outputs_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\n[info] Wrote outputs and latency summary: {output_dir / 'outputs_manifest.json'}")
    except (FileNotFoundError, ImportError, ValueError, RuntimeError) as error:
        print(f"[error] {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
