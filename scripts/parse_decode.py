#!/usr/bin/env python3
"""
把一次实验目录下的 metric-*.txt 解析为 metrics.json。

期望文件名:
    metrics/metric-<testset>-<backend>[-tag].txt   (backend ∈ {pt, onnx})
    (在文件名里区分 testset 与 backend;'pt' = icefall decode.py;'onnx' = sherpa-onnx)

icefall decode.py 产出的 metric 文件格式(每个 keyword 一个块):

    all:
            TP        FP        FN        TN
            263       1         44        N
            Accuracy: 0.866
            Precision: 0.996
            Recall(PPR): 0.857
            FPR: 0.000
            F1: 0.922

sherpa-onnx 评测脚本如果输出同样的 6 行格式,也能被解析。若你的 sherpa-onnx
评测脚本输出格式不同,可参考 _parse_metric_text() 自行扩展。

用法:
    python scripts/parse_decode.py runs/exp001_baseline
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from parse_perf import collect_perf  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent


# --- 极简 YAML 子集解析(避免引入 PyYAML 依赖) ---------------------------

def _load_simple_yaml(path: Path) -> dict:
    """只支持 2 级缩进、键值对、行内注释、字符串/数字/布尔。够用即可。"""
    data: Dict = {}
    stack: List[tuple] = [(0, data)]  # (indent, dict)
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            # 去掉行内注释
            line_nocomment = re.sub(r"\s+#.*$", "", line)
            indent = len(line_nocomment) - len(line_nocomment.lstrip(" "))
            content = line_nocomment.strip()
            if ":" not in content:
                continue
            key, _, val = content.partition(":")
            key = key.strip()
            val = val.strip()
            # 调整父节点
            while stack and indent < stack[-1][0]:
                stack.pop()
            parent = stack[-1][1]
            if val == "":
                node: Dict = {}
                parent[key] = node
                stack.append((indent + 2, node))
            else:
                parent[key] = _coerce(val)
    return data


def _coerce(v: str):
    if v.startswith('"') and v.endswith('"'):
        return v[1:-1]
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v


# --- metric-*.txt 解析 -------------------------------------------------------

_HEADER_RE = re.compile(r"^([A-Za-z0-9 _'\-]+?):\s*$")
_COUNTS_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+(\S+)\s*$")
_FLOAT_RE = re.compile(r"^\s*(Accuracy|Precision|Recall\(PPR\)|FPR|F1)\s*:\s*([0-9.]+)")


def _parse_metric_text(text: str) -> "OrderedDict[str, dict]":
    """返回 {keyword -> {TP,FP,FN,TN,accuracy,precision,recall,fpr,f1}}。"""
    blocks: "OrderedDict[str, dict]" = OrderedDict()
    cur_key: Optional[str] = None
    cur: Dict = {}
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        m = _HEADER_RE.match(line)
        if m and not line.startswith("\t") and not line.startswith(" "):
            # 新块
            if cur_key is not None:
                blocks[cur_key] = cur
            cur_key = m.group(1).strip()
            cur = {}
            i += 1
            continue
        # TP/FP/FN/TN 表头行(包含字面 'TP'),跳过
        if "TP" in line and "FP" in line and "FN" in line and "TN" in line:
            # 下一行是计数
            if i + 1 < len(lines):
                cm = _COUNTS_RE.match(lines[i + 1])
                if cm:
                    cur["TP"] = int(cm.group(1))
                    cur["FP"] = int(cm.group(2))
                    cur["FN"] = int(cm.group(3))
                    tn = cm.group(4)
                    cur["TN"] = int(tn) if tn.isdigit() else None
                i += 2
                continue
        fm = _FLOAT_RE.match(line)
        if fm:
            name = fm.group(1).lower().replace("(ppr)", "")
            cur[name] = float(fm.group(2))
        i += 1
    if cur_key is not None:
        blocks[cur_key] = cur
    return blocks


# --- 主流程 -----------------------------------------------------------------

def _discover_metric_files(metrics_dir: Path) -> List[Path]:
    """匹配 metric-{testset}-{backend}.txt;允许 metric*.txt 兜底。"""
    files = sorted(metrics_dir.glob("metric-*.txt"))
    return files


_KNOWN_BACKENDS = {"pt", "torch", "pytorch", "onnx", "sherpa", "sherpa_onnx"}


def _split_name(path: Path) -> dict:
    """从 metric-<testset>-<backend>[-tag...].txt 推断 testset/backend。

    约定格式: metric-<testset>-<backend>[-extra...].txt
      - testset 可以是任意名(small/large/mywakeword/...)
      - backend 是 pt / onnx 之一(可选别名:torch/pytorch/sherpa/sherpa_onnx)
      - 后面是可选的补充标记(如阈值 t0.35),汇入 backend 后缀

    例:
      metric-small-pt.txt              -> testset=small,       backend=pt
      metric-mywakeword-onnx.txt       -> testset=mywakeword,  backend=onnx
      metric-car_noise-onnx-t0.35.txt  -> testset=car_noise,   backend=onnx-t0.35
    """
    stem = path.stem
    parts = stem.split("-")
    if len(parts) < 3 or parts[0] != "metric":
        return {"testset": stem, "backend": "pt"}
    testset = parts[1]
    backend_raw = parts[2].lower()
    backend_norm = {
        "pt": "pt", "torch": "pt", "pytorch": "pt",
        "onnx": "onnx", "sherpa": "onnx", "sherpa_onnx": "onnx",
    }.get(backend_raw, backend_raw)
    # 额外后缀(如 t0.35) 汇入 backend
    if len(parts) > 3:
        backend_norm = "-".join([backend_norm] + parts[3:])
    return {"testset": testset, "backend": backend_norm}


def parse_experiment(exp_dir: Path) -> dict:
    cfg_path = exp_dir / "config.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"找不到 config.yaml:{cfg_path}")
    cfg = _load_simple_yaml(cfg_path)

    metrics_dir = exp_dir / "metrics"
    if not metrics_dir.exists():
        raise SystemExit(f"找不到 metrics/ 目录:{metrics_dir}")

    eval_cfg = cfg.get("eval", {})
    # 优先读字典 eval.negative_hours;未命中时向后兼容 negative_hours_<testset>
    hours_dict = dict(eval_cfg.get("negative_hours") or {})
    for k, v in eval_cfg.items():
        if isinstance(k, str) and k.startswith("negative_hours_") and k != "negative_hours":
            ts = k[len("negative_hours_"):]
            hours_dict.setdefault(ts, v)

    def _hours_for(testset: str):
        v = hours_dict.get(testset)
        return float(v) if v is not None else None

    runs = []
    for f in _discover_metric_files(metrics_dir):
        meta = _split_name(f)
        blocks = _parse_metric_text(f.read_text(encoding="utf-8", errors="ignore"))
        hours = _hours_for(meta["testset"])
        per_kw = []
        all_block = None
        for kw, d in blocks.items():
            entry = {
                "keyword": kw,
                "TP": d.get("TP"),
                "FP": d.get("FP"),
                "FN": d.get("FN"),
                "TN": d.get("TN"),
                "recall": d.get("recall"),
                "precision": d.get("precision"),
                "accuracy": d.get("accuracy"),
                "fpr": d.get("fpr"),
                "f1": d.get("f1"),
            }
            if hours and entry.get("FP") is not None:
                entry["fa_per_hour"] = round(entry["FP"] / hours, 4)
            else:
                entry["fa_per_hour"] = None
            if kw.lower() == "all":
                all_block = entry
            else:
                per_kw.append(entry)
        runs.append({
            "file": f.name,
            "testset": meta["testset"],
            "backend": meta["backend"],
            "negative_hours": hours,
            "summary": all_block,
            "per_keyword": per_kw,
        })

    perf_runs = collect_perf(exp_dir)

    result = {
        "exp_id": cfg.get("meta", {}).get("exp_id", exp_dir.name.split("_", 1)[0]),
        "name": cfg.get("meta", {}).get("name", ""),
        "date": cfg.get("meta", {}).get("date", ""),
        "variable": cfg.get("meta", {}).get("variable", ""),
        "value": cfg.get("meta", {}).get("value", ""),
        "notes": cfg.get("meta", {}).get("notes", ""),
        "config": cfg,
        "runs": runs,
        "perf_runs": perf_runs,
    }

    out = exp_dir / "metrics.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印摘要
    print(f"已写入 {out}")
    for r in runs:
        s = r["summary"] or {}
        print(f"  - {r['file']:35s} testset={r['testset']:5s} backend={r['backend']:4s}"
              f"  recall={s.get('recall')}  FA/h={s.get('fa_per_hour')}")
    if not runs:
        print("  (未发现 metric-*.txt;请放入 metrics/ 后重试)")
    if perf_runs:
        print(f"顺便收集了 {len(perf_runs)} 个 perf-*.json:")
        for p in perf_runs:
            s = p["summary"]
            extra = f" c={s['concurrency']}" if s.get('concurrency') else (
                    f" bs={s['batch_size']}" if s.get('batch_size') else "")
            print(f"  - {p['file']:35s} mode={s['mode']:10s}{extra}"
                  f" xRT={s.get('throughput_xrt')}"
                  f" p95={s.get('latency_p95')}s")
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("exp_dir", help="实验目录,如 runs/exp001_baseline")
    args = ap.parse_args()
    exp_dir = Path(args.exp_dir).resolve()
    if not exp_dir.is_absolute():
        exp_dir = (ROOT / args.exp_dir).resolve()
    if not exp_dir.exists():
        sys.exit(f"目录不存在:{exp_dir}")
    parse_experiment(exp_dir)


if __name__ == "__main__":
    main()
