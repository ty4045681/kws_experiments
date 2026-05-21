#!/usr/bin/env python3
"""
把一次实验的 metrics.json 写入三张总表:
  - registry.csv     宽表,一行一个实验
  - per_command.csv  长表,一行 = (exp_id, testset, backend, keyword, metric, value)
  - per_perf.csv     长表,一行 = (exp_id, scene, backend, mode, 各 perf 指标)

幂等:同一个 exp_id 重复运行会更新而不重复追加。
没有指定实验目录时,扫描 runs/*/metrics.json 全量重建。

用法:
    python scripts/update_registry.py runs/exp001_baseline
    python scripts/update_registry.py --rebuild
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List


ROOT = Path(__file__).resolve().parent.parent
RUNS_DIR = ROOT / "runs"
REGISTRY = ROOT / "registry.csv"
PER_CMD = ROOT / "per_command.csv"
PER_PERF = ROOT / "per_perf.csv"


# 这些超参字段会从 config.yaml 平铺到 registry.csv 里(命名空间.字段)
FLAT_CONFIG_FIELDS = [
    ("train", "stage"),
    ("train", "base_model"),
    ("train", "subset"),
    ("train", "bpe_size"),
    ("train", "num_epochs"),
    ("train", "base_lr"),
    ("train", "lr_epochs"),
    ("train", "lr_batches"),
    ("train", "max_duration"),
    ("train", "causal"),
    ("decode", "epoch"),
    ("decode", "avg"),
    ("decode", "chunk_size"),
    ("decode", "left_context_frames"),
    ("decode", "keywords_score"),
    ("decode", "keywords_threshold"),
    ("onnx", "exported"),
    ("onnx", "quant"),
    ("onnx", "chunk_size"),
    ("onnx", "left_context_frames"),
]


def _summary_cols(j: dict) -> Dict[str, object]:
    """从 metrics.json 抽出 summary,展平为 recall_{testset}_{backend} 等列。"""
    out: Dict[str, object] = {}
    for run in j.get("runs", []):
        testset = run["testset"]
        backend = run["backend"]
        s = run.get("summary") or {}
        prefix = f"{backend}_{testset}"
        out[f"recall_{prefix}"] = s.get("recall")
        out[f"precision_{prefix}"] = s.get("precision")
        out[f"f1_{prefix}"] = s.get("f1")
        out[f"fp_{prefix}"] = s.get("FP")
        out[f"fn_{prefix}"] = s.get("FN")
        out[f"tp_{prefix}"] = s.get("TP")
        out[f"fa_per_hour_{prefix}"] = s.get("fa_per_hour")

    # 性能指标展平:对每个 perf-*.json 出 4 列
    for p in j.get("perf_runs", []):
        scene = p.get("scene", "")
        backend = p.get("backend", "")
        s = p.get("summary") or {}
        prefix = f"{scene}_{backend}"
        out[f"xrt_{prefix}"] = s.get("throughput_xrt")
        out[f"cps_{prefix}"] = s.get("throughput_cps")
        out[f"latp50_{prefix}"] = s.get("latency_p50")
        out[f"latp95_{prefix}"] = s.get("latency_p95")
    return out


def _flatten_config(cfg: dict) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for ns, key in FLAT_CONFIG_FIELDS:
        out[f"{ns}.{key}"] = cfg.get(ns, {}).get(key)
    return out


def _row_for_experiment(j: dict) -> Dict[str, object]:
    row: Dict[str, object] = {
        "exp_id": j.get("exp_id", ""),
        "name": j.get("name", ""),
        "date": j.get("date", ""),
        "variable": j.get("variable", ""),
        "value": j.get("value", ""),
        "notes": j.get("notes", ""),
    }
    row.update(_flatten_config(j.get("config", {})))
    row.update(_summary_cols(j))
    return row


def _per_perf_rows(j: dict) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    exp_id = j.get("exp_id", "")
    for p in j.get("perf_runs", []):
        s = p.get("summary") or {}
        rows.append({
            "exp_id": exp_id,
            "scene": p.get("scene"),
            "backend": p.get("backend"),
            "tag": p.get("tag", ""),
            "mode": s.get("mode"),
            "concurrency": s.get("concurrency"),
            "batch_size": s.get("batch_size"),
            "throughput_xrt": s.get("throughput_xrt"),
            "throughput_cps": s.get("throughput_cps"),
            "latency_p50": s.get("latency_p50"),
            "latency_p95": s.get("latency_p95"),
            "latency_p99": s.get("latency_p99"),
            "rtf_mean": s.get("rtf_mean"),
            "file": p.get("file"),
        })
    return rows


def _per_command_rows(j: dict) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    exp_id = j.get("exp_id", "")
    for run in j.get("runs", []):
        testset = run["testset"]
        backend = run["backend"]
        for kw in run.get("per_keyword", []):
            base = {
                "exp_id": exp_id,
                "testset": testset,
                "backend": backend,
                "keyword": kw.get("keyword"),
                "TP": kw.get("TP"),
                "FP": kw.get("FP"),
                "FN": kw.get("FN"),
                "TN": kw.get("TN"),
                "recall": kw.get("recall"),
                "precision": kw.get("precision"),
                "f1": kw.get("f1"),
                "fpr": kw.get("fpr"),
                "fa_per_hour": kw.get("fa_per_hour"),
            }
            rows.append(base)
    return rows


def _union_keys(rows: Iterable[Dict[str, object]]) -> List[str]:
    return list(dict.fromkeys(k for r in rows for k in r.keys()))


def _read_csv(path: Path) -> List[Dict[str, object]]:
    try:
        with path.open("r", encoding="utf-8", newline="") as f:
            return list(csv.DictReader(f))
    except FileNotFoundError:
        return []


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        # 空表保留(写一个空文件)而不是删除——避免幂等重跑时把别人的 CSV 抹了
        path.write_text("", encoding="utf-8")
        return
    fieldnames = _union_keys(rows)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if (v := r.get(k)) is None else v) for k in fieldnames})


def _load_all_metrics() -> List[dict]:
    out = []
    if not RUNS_DIR.exists():
        return out
    for p in sorted(RUNS_DIR.iterdir()):
        mj = p / "metrics.json"
        if mj.exists():
            out.append(json.loads(mj.read_text(encoding="utf-8")))
    return out


def update_one(exp_dir: Path) -> None:
    mj = exp_dir / "metrics.json"
    if not mj.exists():
        raise SystemExit(f"先跑 parse_decode.py:缺少 {mj}")
    new = json.loads(mj.read_text(encoding="utf-8"))
    exp_id = new.get("exp_id", "")

    # 宽表
    reg = _read_csv(REGISTRY)
    reg = [r for r in reg if r.get("exp_id") != exp_id]
    reg.append(_row_for_experiment(new))
    reg.sort(key=lambda r: r.get("exp_id", ""))
    _write_csv(REGISTRY, reg)

    # 长表
    pc = _read_csv(PER_CMD)
    pc = [r for r in pc if r.get("exp_id") != exp_id]
    pc.extend(_per_command_rows(new))
    pc.sort(key=lambda r: (r.get("exp_id", ""), r.get("testset", ""), r.get("backend", ""), r.get("keyword", "")))
    _write_csv(PER_CMD, pc)

    # 长表 - per_perf
    pp = _read_csv(PER_PERF)
    pp = [r for r in pp if r.get("exp_id") != exp_id]
    pp.extend(_per_perf_rows(new))
    pp.sort(key=lambda r: (r.get("exp_id", ""), r.get("scene", ""), r.get("backend", "")))
    _write_csv(PER_PERF, pp)

    print(f"已更新 {REGISTRY.name}({len(reg)} 行)、"
          f"{PER_CMD.name}({len(pc)} 行)、{PER_PERF.name}({len(pp)} 行)")


def rebuild_all() -> None:
    metrics = _load_all_metrics()
    reg = [_row_for_experiment(m) for m in metrics]
    reg.sort(key=lambda r: r.get("exp_id", ""))
    _write_csv(REGISTRY, reg)
    pc: List[Dict[str, object]] = []
    pp: List[Dict[str, object]] = []
    for m in metrics:
        pc.extend(_per_command_rows(m))
        pp.extend(_per_perf_rows(m))
    pc.sort(key=lambda r: (r.get("exp_id", ""), r.get("testset", ""), r.get("backend", ""), r.get("keyword", "")))
    pp.sort(key=lambda r: (r.get("exp_id", ""), r.get("scene", ""), r.get("backend", "")))
    _write_csv(PER_CMD, pc)
    _write_csv(PER_PERF, pp)
    print(f"已重建 {REGISTRY.name}({len(reg)} 行)、"
          f"{PER_CMD.name}({len(pc)} 行)、{PER_PERF.name}({len(pp)} 行)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", nargs="?", help="实验目录,留空则配合 --rebuild")
    ap.add_argument("--rebuild", action="store_true", help="忽略 exp_dir,扫描 runs/*/metrics.json 重建")
    args = ap.parse_args()

    if args.rebuild:
        rebuild_all()
        return
    if not args.exp_dir:
        ap.error("必须给出 exp_dir 或使用 --rebuild")
    p = Path(args.exp_dir)
    if not p.is_absolute():
        p = (ROOT / args.exp_dir).resolve()
    update_one(p)


if __name__ == "__main__":
    main()
