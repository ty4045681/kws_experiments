#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把一次实验目录下的 perf-*.json 收集合并到 metrics.json 的 `perf_runs` 字段。

期望文件名:
    metrics/perf-<scene>-<backend>[-tag].json   (backend 一般是 onnx)

可单独跑:
    python scripts/parse_perf.py runs/exp001_baseline

也会被 scripts/parse_decode.py 自动调用,所以一般不需要手动跑。

效果:
    runs/<exp>/metrics.json
        + perf_runs : [
              {file, scene, backend, mode, summary:{...}, raw:{...}},
              ...
          ]
    其中 summary 是为 registry.csv 抽出的扁平指标。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(__file__).resolve().parent.parent


# ─── 从 perf JSON 抽出"上 registry 用"的扁平 summary ─────────────────────

def _flat_summary(payload: dict) -> dict:
    """从 perf JSON 抽 6~8 个关键指标。键命名稳定,便于 update_registry 拼列。"""
    mode = payload.get("mode")
    r = payload.get("result", {}) or {}
    out: Dict[str, Optional[float]] = {
        "mode": mode,
        "throughput_xrt": r.get("throughput_audio_per_wall"),
        "throughput_cps": r.get("throughput_calls_per_sec"),
        "latency_p50": None,
        "latency_p95": None,
        "latency_p99": None,
        "rtf_mean": None,
        "concurrency": None,
        "batch_size": None,
    }
    lat = r.get("latency_seconds") or r.get("per_call_latency_seconds") or {}
    out["latency_p50"] = lat.get("p50")
    out["latency_p95"] = lat.get("p95")
    out["latency_p99"] = lat.get("p99")

    if mode == "single":
        rtf = r.get("rtf") or {}
        out["rtf_mean"] = rtf.get("mean")
    elif mode == "concurrent":
        rtf = r.get("rtf_per_stream") or {}
        out["rtf_mean"] = rtf.get("mean")
        out["concurrency"] = r.get("concurrency")
    elif mode == "batch":
        rtf = r.get("batch_rtf") or {}
        out["rtf_mean"] = rtf.get("mean")
        out["batch_size"] = r.get("batch_size")
    return out


# ─── 从文件名拆 scene / backend ─────────────────────────────────────────

def _split_perf_name(path: Path) -> dict:
    """perf-<scene>-<backend>[-tag].json -> {scene, backend}"""
    stem = path.stem  # 不含 .json
    parts = stem.split("-")
    if len(parts) < 3 or parts[0] != "perf":
        return {"scene": stem, "backend": "onnx"}
    # scene 允许是 single_cpu1t 这种带下划线的,占一段
    scene = parts[1]
    backend = parts[2]
    # 多余段并入 backend(如 perf-single_cpu1t-onnx-int8.json -> backend=onnx-int8)
    if len(parts) > 3:
        backend = "-".join([backend] + parts[3:])
    return {"scene": scene, "backend": backend}


# ─── 主收集函数:扫一个实验目录 ────────────────────────────────────────

def collect_perf(exp_dir: Path) -> List[dict]:
    """扫 exp_dir/metrics/perf-*.json,返回 perf_runs 列表。"""
    metrics_dir = exp_dir / "metrics"
    if not metrics_dir.exists():
        return []
    runs = []
    for f in sorted(metrics_dir.glob("perf-*.json")):
        try:
            payload = json.loads(f.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  [warn] 跳过 {f.name}: {e}", file=sys.stderr)
            continue
        meta = _split_perf_name(f)
        runs.append({
            "file": f.name,
            "scene": payload.get("scene", meta["scene"]),
            "backend": payload.get("backend", meta["backend"]),
            "tag": payload.get("tag", ""),
            "mode": payload.get("mode"),
            "summary": _flat_summary(payload),
            "env": payload.get("env", {}),
            "model": payload.get("model", {}),
            "raw": payload.get("result", {}),
        })
    return runs


def merge_into_metrics_json(exp_dir: Path) -> int:
    """读 exp_dir/metrics.json,把 perf_runs 字段覆盖更新进去。"""
    mj = exp_dir / "metrics.json"
    if not mj.exists():
        print(f"[warn] {mj} 不存在,请先跑 parse_decode.py。"
              f"现在只把 perf_runs 单独写出。", file=sys.stderr)
        data: dict = {"exp_id": exp_dir.name.split("_", 1)[0], "runs": []}
    else:
        data = json.loads(mj.read_text(encoding="utf-8"))
    perf_runs = collect_perf(exp_dir)
    data["perf_runs"] = perf_runs
    mj.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return len(perf_runs)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", help="实验目录,如 runs/exp001_baseline")
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_absolute():
        exp_dir = (ROOT / args.exp_dir).resolve()
    if not exp_dir.exists():
        sys.exit(f"目录不存在:{exp_dir}")
    n = merge_into_metrics_json(exp_dir)
    print(f"已合并 {n} 个 perf-*.json -> {exp_dir/'metrics.json'}")
    if n:
        data = json.loads((exp_dir / "metrics.json").read_text(encoding="utf-8"))
        for p in data.get("perf_runs", []):
            s = p["summary"]
            extra = ""
            if s.get("concurrency"):
                extra = f"  c={s['concurrency']}"
            elif s.get("batch_size"):
                extra = f"  bs={s['batch_size']}"
            print(f"  - {p['file']:35s} mode={s['mode']:10s}{extra}"
                  f"  xRT={s.get('throughput_xrt')}"
                  f"  p95={s.get('latency_p95')}s")


if __name__ == "__main__":
    main()
