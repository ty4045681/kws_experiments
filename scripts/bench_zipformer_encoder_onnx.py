#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
用 ONNX Runtime 单独对 sherpa-onnx Zipformer encoder 做前向传播时延测试。

绑定到 CPU[0]、单线程、CPU 推理，通过随机输入测 encoder 本身的推理延迟。

典型用法:
    python scripts/bench_zipformer_encoder_onnx.py \
        --encoder model/encoder-epoch-12-avg-2-chunk-16-left-128.onnx \
        --batch-size 1 \
        --chunk-frames 16 \
        --feature-dim 80 \
        --warmup 50 \
        --iterations 500 \
        --output metrics/perf-encoder_single-onnx.json

输出:
    - 终端打印 latency 统计(p50/p90/p95/p99/mean/std)
    - 如指定 --output,写与 sherpa_perf 兼容的 perf JSON,可被 scripts/parse_perf.py 解析
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

ort = None  # 延迟导入,让 --help 在不装 onnxruntime 时也能用


def _ensure_ort():
    global ort
    if ort is not None:
        return ort
    try:
        import onnxruntime as _ort
    except ImportError:
        print("[error] 需要先安装 onnxruntime: pip install onnxruntime", file=sys.stderr)
        raise
    ort = _ort
    return ort


def _bind_cpu0() -> None:
    """尽量把当前进程绑定到 CPU 0;非 Linux 平台仅打印警告。"""
    if platform.system() == "Linux":
        try:
            os.sched_setaffinity(0, {0})
            print("[info] 已绑定到 CPU 0")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] 绑定 CPU 0 失败: {exc}", file=sys.stderr)
    else:
        print(
            f"[warn] {platform.system()} 不支持 os.sched_setaffinity,\n"
            "      CPU 绑定请在 Linux 下用 taskset -c 0 python ... 启动",
            file=sys.stderr,
        )


def _build_session(encoder_path: str, intra_op: int, inter_op: int):
    """构造单线程 CPU 推理的 ONNX Runtime session。"""
    _ensure_ort()
    sess_opts = ort.SessionOptions()
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opts.intra_op_num_threads = intra_op
    sess_opts.inter_op_num_threads = inter_op
    sess_opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(
        encoder_path,
        sess_options=sess_opts,
        providers=["CPUExecutionProvider"],
    )


def _inspect_inputs(session) -> None:
    """打印模型输入名、形状、数据类型，方便核对。"""
    print("[info] encoder 输入信息:")
    for inp in session.get_inputs():
        print(f"    {inp.name:30s} shape={inp.shape}, dtype={inp.type}")


def _resolve_shape(
    raw_shape: List[Any],
    name: str,
    batch_size: int,
    chunk_frames: int,
    feature_dim: int,
) -> List[int]:
    """把 ONNX 输入 shape 里的动态维度(字符串或 0/'?')替换为具体数值。"""
    out: List[int] = []
    for idx, dim in enumerate(raw_shape):
        try:
            v = int(dim)  # 处理 int / 数字字符串 / DimensionValue 等
            if v > 0:
                out.append(v)
                continue
        except (TypeError, ValueError):
            pass
        # 动态维度:按常见约定推断
        if name.lower() in ("x", "features", "feature", "feats"):
            if idx == 0:
                out.append(batch_size)
            elif idx == 1:
                out.append(chunk_frames)
            elif idx == 2:
                out.append(feature_dim)
            else:
                out.append(1)
        elif name.lower() in ("x_lens", "lengths", "lens"):
            out.append(batch_size)
        else:
            # states / cache: 第 0 维一般是 batch,其余按 1 或实际值兜底
            if idx == 0:
                out.append(batch_size)
            else:
                out.append(1)
    return out


def _numpy_type(onnx_type: str) -> np.dtype:
    mapping = {
        "tensor(float)": np.float32,
        "tensor(float16)": np.float16,
        "tensor(double)": np.float64,
        "tensor(int32)": np.int32,
        "tensor(int64)": np.int64,
        "tensor(uint8)": np.uint8,
        "tensor(bool)": bool,
    }
    return mapping.get(onnx_type, np.float32)


def _make_inputs(
    session,
    batch_size: int,
    chunk_frames: int,
    feature_dim: int,
    state_fill: str,
) -> Dict[str, np.ndarray]:
    """为 encoder 构造随机或零填充的输入字典。"""
    inputs: Dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = _resolve_shape(inp.shape, inp.name, batch_size, chunk_frames, feature_dim)
        dtype = _numpy_type(inp.type)

        is_feature = inp.name.lower() in ("x", "features", "feature", "feats")
        is_length = inp.name.lower() in ("x_lens", "lengths", "lens")

        if is_length:
            arr = np.full(shape, chunk_frames, dtype=dtype)
        elif is_feature:
            arr = np.random.randn(*shape).astype(dtype)
        elif state_fill == "random":
            arr = np.random.randn(*shape).astype(dtype)
        else:
            # states/cache 默认用 zeros,与 streaming 初始状态一致
            arr = np.zeros(shape, dtype=dtype)

        inputs[inp.name] = arr
    return inputs


def _format_latency_ms(times_ms: List[float]) -> Dict[str, float]:
    """把毫秒列表整理为统计摘要。"""
    return {
        "mean": statistics.mean(times_ms),
        "std": statistics.stdev(times_ms) if len(times_ms) > 1 else 0.0,
        "min": min(times_ms),
        "max": max(times_ms),
        "p50": float(np.percentile(times_ms, 50)),
        "p90": float(np.percentile(times_ms, 90)),
        "p95": float(np.percentile(times_ms, 95)),
        "p99": float(np.percentile(times_ms, 99)),
    }


def _print_summary(summary: Dict[str, Any]) -> None:
    print("\n[result] encoder 推理时延统计(ms):")
    for k in ("mean", "std", "min", "p50", "p90", "p95", "p99", "max"):
        print(f"    {k:6s}: {summary['latency_ms'][k]:.3f}")
    print(f"\n    总调用次数: {summary['iterations']}")
    print(f"    环境: {summary['platform']}")


def _write_perf_json(
    out_path: Path,
    summary: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    """写成 sherpa_perf 风格的 perf JSON,便于被 scripts/parse_perf.py 收纳。"""
    payload = {
        "file": str(out_path),
        "scene": "encoder_single",
        "backend": "onnx",
        "mode": "single",
        "result": {
            "latency_ms": summary["latency_ms"],
            "per_call_latency_seconds": {
                k: v / 1000.0
                for k, v in summary["latency_ms"].items()
            },
            "throughput_calls_per_sec": 1000.0 / summary["latency_ms"]["mean"],
            "batch_size": args.batch_size,
            "chunk_frames": args.chunk_frames,
            "feature_dim": args.feature_dim,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "intra_op_num_threads": args.intra_op_num_threads,
            "inter_op_num_threads": args.inter_op_num_threads,
            "provider": "CPUExecutionProvider",
        },
        "config": {
            "encoder": str(Path(args.encoder).resolve()),
            "batch_size": args.batch_size,
            "chunk_frames": args.chunk_frames,
            "feature_dim": args.feature_dim,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "state_fill": args.state_fill,
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[info] 已写入 perf JSON: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    ap.add_argument("--encoder", required=True, help="encoder ONNX 文件路径")
    ap.add_argument("--batch-size", type=int, default=1, help="batch 维度大小(默认 1)")
    ap.add_argument("--chunk-frames", type=int, default=16, help="x 输入的时间帧长(默认 16)")
    ap.add_argument("--feature-dim", type=int, default=80, help="x 输入的特征维度(默认 80)")
    ap.add_argument("--state-fill", choices=["zeros", "random"], default="zeros",
                    help="states/cache 类输入的填充方式(默认 zeros)")

    ap.add_argument("--warmup", type=int, default=50, help="warmup 轮次(默认 50)")
    ap.add_argument("--iterations", type=int, default=500, help="正式测试轮次(默认 500)")

    ap.add_argument("--intra-op-num-threads", type=int, default=1,
                    help="ONNX Runtime intra_op 线程数(默认 1)")
    ap.add_argument("--inter-op-num-threads", type=int, default=1,
                    help="ONNX Runtime inter_op 线程数(默认 1)")
    ap.add_argument("--no-cpu-bind", action="store_true",
                    help="跳过 CPU[0] 绑定(某些平台不支持)")

    ap.add_argument("--output", help="输出 perf JSON 路径,例如 metrics/perf-encoder_single-onnx.json")
    ap.add_argument("--seed", type=int, default=42, help="随机种子(默认 42)")

    args = ap.parse_args()

    if not Path(args.encoder).is_file():
        print(f"[error] 找不到 encoder: {args.encoder}", file=sys.stderr)
        sys.exit(1)

    np.random.seed(args.seed)
    _ensure_ort()

    # 限制 OpenMP / MKL 等多线程后端,确保总体是单线程行为
    os.environ.setdefault("OMP_NUM_THREADS", str(args.intra_op_num_threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.intra_op_num_threads))

    if not args.no_cpu_bind:
        _bind_cpu0()

    session = _build_session(
        args.encoder,
        args.intra_op_num_threads,
        args.inter_op_num_threads,
    )
    _inspect_inputs(session)

    inputs = _make_inputs(
        session,
        args.batch_size,
        args.chunk_frames,
        args.feature_dim,
        args.state_fill,
    )
    print(f"[info] 输入 x shape: {inputs.get('x', next(iter(inputs.values()))).shape}, "
          f"iterations={args.iterations}, warmup={args.warmup}")

    # warmup
    print(f"[info] warmup {args.warmup} 轮...")
    for _ in range(args.warmup):
        session.run(None, inputs)

    # benchmark
    print(f"[info] 正式测试 {args.iterations} 轮...")
    times_ms: List[float] = []
    for _ in range(args.iterations):
        t0 = time.perf_counter()
        session.run(None, inputs)
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    latency_ms = _format_latency_ms(times_ms)
    summary = {
        "latency_ms": latency_ms,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "platform": f"{platform.system()} {platform.machine()}",
        "provider": "CPUExecutionProvider",
    }

    _print_summary(summary)

    if args.output:
        _write_perf_json(Path(args.output), summary, args)


if __name__ == "__main__":
    main()
