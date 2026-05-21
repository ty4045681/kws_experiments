#!/usr/bin/env python3
"""
从 registry.csv / per_command.csv / per_perf.csv 生成 REPORT.md。

设计原则(2026-05-20 重构):
  - 不再用 registry 的宽表做主总览(列太多、太稀疏)。
  - 准确率主总览以 per_command.csv(长表)聚合,每行只展示真有数据的格子。
  - 每个 testset 单独出一张"实验 × backend/tag"小表,自动剪掉全空列。
  - backend 形如 `onnx-tXX` 视为"阈值扫描" tag,自动抽出来出阈值小节。
  - 性能(perf)章节维持原样:总览 / 最佳 / 详细。

用法:
    python scripts/build_report.py
"""

from __future__ import annotations

import csv
import datetime
import re
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


ROOT = Path(__file__).resolve().parent.parent
REGISTRY = ROOT / "registry.csv"
PER_CMD = ROOT / "per_command.csv"
PER_PERF = ROOT / "per_perf.csv"
OUT = ROOT / "REPORT.md"

_PERF_NDIGITS = 3


# ─── 通用工具 ───────────────────────────────────────────────────────────

def _read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _fmt(v: Any, ndigits: int = 3) -> str:
    if v is None or v == "" or (isinstance(v, float) and v != v):  # NaN
        return "—"
    try:
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return str(v)


def _to_md_table(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    out = ["| " + " | ".join(rows[0]) + " |",
           "|" + "|".join(["---"] * len(rows[0])) + "|"]
    for r in rows[1:]:
        out.append("| " + " | ".join(r) + " |")
    return "\n".join(out)


def _split_backend_tag(backend: str) -> Tuple[str, str]:
    """`onnx-t0.20` -> ('onnx', '0.20');`onnx` -> ('onnx', '');`onnx-int8` -> ('onnx', 'int8')。"""
    if not backend:
        return ("", "")
    parts = backend.split("-", 1)
    if len(parts) == 1:
        return (parts[0], "")
    base, tail = parts
    m = re.match(r"^t(.+)$", tail)
    if m:
        return (base, m.group(1))
    return (base, tail)


# ─── 准确率 · 主总览(基于 per_command.csv,长表聚合) ──────────────────

def _aggregate_keywords(pc_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """把 per_command 按 (exp_id, testset, backend) 聚合成"all-keyword 合计行"。
    若已存在 keyword=='all',直接用;否则按 TP/FP/FN/TN 求和算 recall/precision。
    """
    by_key: Dict[Tuple[str, str, str], List[Dict[str, str]]] = defaultdict(list)
    for r in pc_rows:
        by_key[(r.get("exp_id", ""), r.get("testset", ""), r.get("backend", ""))].append(r)

    out: List[Dict[str, Any]] = []
    for (eid, ts, bk), rs in by_key.items():
        all_row = next((r for r in rs if (r.get("keyword") or "").lower() == "all"), None)
        if all_row:
            base, tag = _split_backend_tag(bk)
            out.append({
                "exp_id": eid, "testset": ts, "backend_raw": bk,
                "backend_base": base, "tag": tag,
                "keyword": "all",
                "tp": float(all_row.get("TP") or 0),
                "fp": float(all_row.get("FP") or 0),
                "fn": float(all_row.get("FN") or 0),
                "tn": float(all_row.get("TN") or 0),
                "recall": float(all_row["recall"]) if all_row.get("recall") else None,
                "precision": float(all_row["precision"]) if all_row.get("precision") else None,
                "f1": float(all_row["f1"]) if all_row.get("f1") else None,
                "fpr": float(all_row["fpr"]) if all_row.get("fpr") else None,
                "fa_per_hour": float(all_row["fa_per_hour"]) if all_row.get("fa_per_hour") else None,
                "n_keywords": len(rs),
            })
            continue
        # 没有 'all' 行,按计数加总
        tp = sum(float(r.get("TP") or 0) for r in rs)
        fp = sum(float(r.get("FP") or 0) for r in rs)
        fn = sum(float(r.get("FN") or 0) for r in rs)
        tn = sum(float(r.get("TN") or 0) for r in rs)
        recall = tp / (tp + fn) if (tp + fn) > 0 else None
        precision = tp / (tp + fp) if (tp + fp) > 0 else None
        f1 = (2 * precision * recall / (precision + recall)
              if precision and recall else None)
        # fa_per_hour 不能简单求和,跳过;若任一非空,取平均
        fa_vals = [float(r["fa_per_hour"]) for r in rs
                   if r.get("fa_per_hour") not in (None, "")]
        fa = sum(fa_vals) / len(fa_vals) if fa_vals else None
        base, tag = _split_backend_tag(bk)
        out.append({
            "exp_id": eid, "testset": ts, "backend_raw": bk,
            "backend_base": base, "tag": tag,
            "keyword": "(aggregated)",
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision, "f1": f1,
            "fpr": None, "fa_per_hour": fa,
            "n_keywords": len(rs),
        })
    return out


def _long_summary_table(agg: List[Dict[str, Any]]) -> str:
    """长表总览:一行一个 (exp, testset, backend[, tag])。密实、零 NaN 列。"""
    if not agg:
        return ""
    has_tag = any(a["tag"] for a in agg)
    has_fa = any(a.get("fa_per_hour") is not None for a in agg)
    header = ["exp_id", "testset", "backend"]
    if has_tag:
        header.append("tag")
    header += ["recall", "precision", "F1", "TP", "FP", "FN"]
    if has_fa:
        header.append("FA/h")
    rows = [header]
    # 排序:exp_id, testset, backend_base, tag 数值化
    def sort_key(a):
        try: tag_n = float(a["tag"]) if a["tag"] else -1
        except Exception: tag_n = 0
        return (a["exp_id"], a["testset"], a["backend_base"], tag_n)
    for a in sorted(agg, key=sort_key):
        row = [a["exp_id"], a["testset"], a["backend_base"]]
        if has_tag:
            row.append(a["tag"] or "—")
        row += [
            _fmt(a.get("recall")), _fmt(a.get("precision")), _fmt(a.get("f1")),
            f"{int(a.get('tp') or 0)}", f"{int(a.get('fp') or 0)}",
            f"{int(a.get('fn') or 0)}",
        ]
        if has_fa:
            row.append(_fmt(a.get("fa_per_hour")))
        rows.append(row)
    return _to_md_table(rows)


# ─── 准确率 · 按 testset 分组(每张表都剪掉全空列) ─────────────────────

def _per_testset_section(agg: List[Dict[str, Any]]) -> str:
    """每个 testset 一张表:exp_id × (backend, tag) 的 recall|precision。"""
    by_ts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in agg:
        by_ts[a["testset"]].append(a)
    if not by_ts:
        return ""
    blocks: List[str] = []
    for ts in sorted(by_ts.keys()):
        rows_data = by_ts[ts]
        # 该 testset 实际出现过的 (backend_base, tag)
        bk_tags = sorted({(r["backend_base"], r["tag"]) for r in rows_data},
                         key=lambda x: (x[0], _safe_tag_sort(x[1])))
        # 该 testset 实际出现过的实验
        exp_ids = sorted({r["exp_id"] for r in rows_data})
        # 构表:行 = exp_id;列 = (backend, tag) 的 R/P/F1
        header = ["exp_id"]
        for bk, tag in bk_tags:
            col_label = f"{bk}" + (f"-t{tag}" if tag else "")
            header += [f"R({col_label})", f"P({col_label})", f"F1({col_label})"]
        rows = [header]
        idx = {(r["exp_id"], r["backend_base"], r["tag"]): r for r in rows_data}
        for eid in exp_ids:
            row = [eid]
            for bk, tag in bk_tags:
                a = idx.get((eid, bk, tag))
                if a is None:
                    row += ["—", "—", "—"]
                else:
                    row += [_fmt(a.get("recall")), _fmt(a.get("precision")), _fmt(a.get("f1"))]
            rows.append(row)
        # 关键:剪掉全 — 的列
        rows = _drop_empty_cols(rows, fill_token="—")
        blocks.append(f"### testset = `{ts}`")
        blocks.append(_to_md_table(rows))
        # 每个 testset 跟一个 TP/FP/FN 计数表(常常更有用)
        cnt_header = ["exp_id"]
        for bk, tag in bk_tags:
            col_label = f"{bk}" + (f"-t{tag}" if tag else "")
            cnt_header += [f"TP({col_label})", f"FP({col_label})", f"FN({col_label})"]
        cnt_rows = [cnt_header]
        for eid in exp_ids:
            row = [eid]
            for bk, tag in bk_tags:
                a = idx.get((eid, bk, tag))
                if a is None:
                    row += ["—", "—", "—"]
                else:
                    row += [str(int(a["tp"])), str(int(a["fp"])), str(int(a["fn"]))]
            cnt_rows.append(row)
        cnt_rows = _drop_empty_cols(cnt_rows, fill_token="—")
        blocks.append("")
        blocks.append(_to_md_table(cnt_rows))
        blocks.append("")
    return "\n".join(blocks).strip()


def _safe_tag_sort(t: str):
    try: return float(t)
    except Exception: return float("inf") if t else -1.0


def _drop_empty_cols(rows: List[List[str]], fill_token: str = "—") -> List[List[str]]:
    if len(rows) < 2:
        return rows
    n_cols = len(rows[0])
    keep_idx = []
    for j in range(n_cols):
        if j == 0:  # 第一列(label)永远留
            keep_idx.append(j); continue
        if any(rows[i][j] != fill_token for i in range(1, len(rows))):
            keep_idx.append(j)
    return [[r[j] for j in keep_idx] for r in rows]


# ─── 准确率 · 阈值扫描小节(检测同实验同 testset 多 tag 的情况) ────────

def _threshold_sweep_section(agg: List[Dict[str, Any]]) -> str:
    """当某个 (exp_id, testset, backend_base) 出现 >=2 个 tag,出"阈值 → 指标"小表。"""
    by_scope: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for a in agg:
        if a["tag"]:
            by_scope[(a["exp_id"], a["testset"], a["backend_base"])].append(a)
    blocks: List[str] = []
    for (eid, ts, bk), rs in by_scope.items():
        if len(rs) < 2:
            continue
        rs = sorted(rs, key=lambda x: _safe_tag_sort(x["tag"]))
        # 用同 (exp, testset, backend_base, tag='') 当 baseline (若存在)
        baseline = next((a for a in agg
                         if a["exp_id"] == eid and a["testset"] == ts
                         and a["backend_base"] == bk and not a["tag"]), None)
        blocks.append(f"### {eid} · {ts} · backend={bk}")
        header = ["tag(threshold)", "recall", "precision", "F1", "TP", "FP", "FN"]
        if any(a.get("fa_per_hour") is not None for a in rs):
            header.append("FA/h")
        rows = [header]
        if baseline:
            row = ["(no-tag)", _fmt(baseline.get("recall")),
                   _fmt(baseline.get("precision")), _fmt(baseline.get("f1")),
                   str(int(baseline["tp"])), str(int(baseline["fp"])), str(int(baseline["fn"]))]
            if "FA/h" in header:
                row.append(_fmt(baseline.get("fa_per_hour")))
            rows.append(row)
        for a in rs:
            row = [a["tag"], _fmt(a.get("recall")), _fmt(a.get("precision")),
                   _fmt(a.get("f1")), str(int(a["tp"])), str(int(a["fp"])),
                   str(int(a["fn"]))]
            if "FA/h" in header:
                row.append(_fmt(a.get("fa_per_hour")))
            rows.append(row)
        # 检查:每行是不是完全一样(暗示评测脚本没真正按 threshold 解码)
        # 检查除 tag 列外是否完全一致(暗示 threshold 没生效)
        body_rows = [tuple(r[1:]) for r in rows[1:] if r[0] != "(no-tag)"]
        identical = len(set(body_rows)) == 1 and len(body_rows) >= 2
        blocks.append(_to_md_table(rows))
        if identical:
            blocks.append(
                "> ⚠️ 所有阈值的结果完全一致 —— 评测脚本可能没按 threshold "
                "重新解码,或 threshold 没传到 sherpa-onnx。请检查 `keywords_threshold` 是否生效。")
        blocks.append("")
    if not blocks:
        return ""
    return "\n".join(blocks).strip()


# ─── 准确率 · 最佳 ─────────────────────────────────────────────────────

def _best_section(agg: List[Dict[str, Any]]) -> str:
    if not agg:
        return ""
    by_ts: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for a in agg:
        if a.get("recall") is not None:
            by_ts[a["testset"]].append(a)
    if not by_ts:
        return ""
    rows = [["testset", "exp_id", "backend", "tag", "recall", "precision", "F1", "FP"]]
    for ts in sorted(by_ts.keys()):
        best = max(by_ts[ts], key=lambda a: (a.get("recall") or -1,
                                              -(a.get("fp") or 0)))
        rows.append([
            ts, best["exp_id"], best["backend_base"], best["tag"] or "—",
            _fmt(best.get("recall")), _fmt(best.get("precision")),
            _fmt(best.get("f1")), str(int(best["fp"])),
        ])
    return ("**每个 testset 的当前最佳(按 Recall;同 Recall 选 FP 更少的):**\n\n"
            + _to_md_table(rows))


# ─── 准确率 · 覆盖矩阵(快速看出哪些组合没测) ──────────────────────────

def _coverage_section(agg: List[Dict[str, Any]]) -> str:
    if not agg:
        return ""
    exp_ids = sorted({a["exp_id"] for a in agg})
    testsets = sorted({a["testset"] for a in agg})
    have: set = {(a["exp_id"], a["testset"]) for a in agg}
    header = ["exp_id"] + testsets
    rows = [header]
    for eid in exp_ids:
        rows.append([eid] + ["✓" if (eid, ts) in have else "·" for ts in testsets])
    return ("_看哪些实验在哪些 testset 上测过(✓=有数据,·=没测)_\n\n"
            + _to_md_table(rows))


# ─── 实验元数据小表(replace 原 registry 总览) ──────────────────────────

def _experiment_meta_section(reg: List[Dict[str, str]]) -> str:
    if not reg:
        return ""
    header = ["exp_id", "name", "variable", "value", "epoch", "avg",
              "keywords_threshold", "notes"]
    rows = [header]
    for r in reg:
        rows.append([
            r.get("exp_id", ""), r.get("name", ""),
            r.get("variable", "") or "—", r.get("value", "") or "—",
            r.get("decode.epoch", "") or "—", r.get("decode.avg", "") or "—",
            r.get("decode.keywords_threshold", "") or "—",
            (r.get("notes") or "").replace("|", "/")[:60],
        ])
    return _to_md_table(rows)


# ─── perf 章节(基本沿用旧实现) ────────────────────────────────────────

def _discover_perf_scopes(reg: List[Dict[str, str]]) -> List[str]:
    seen: List[str] = []
    for r in reg:
        for col in r.keys():
            if col.startswith("xrt_"):
                suf = col[len("xrt_"):]
                if suf and suf not in seen:
                    seen.append(suf)
    return sorted(seen)


def _perf_summary_table(reg: List[Dict[str, str]]) -> str:
    scopes = _discover_perf_scopes(reg)
    if not scopes:
        return ""
    header = ["exp_id", "name"]
    for s in scopes:
        header += [f"xRT({s})", f"cps({s})", f"p95s({s})"]
    rows = [header]
    for r in reg:
        row = [r.get("exp_id", ""), r.get("name", "")]
        for s in scopes:
            row.append(_fmt(r.get(f"xrt_{s}", ""), 2))
            row.append(_fmt(r.get(f"cps_{s}", ""), 2))
            row.append(_fmt(r.get(f"latp95_{s}", ""), _PERF_NDIGITS))
        rows.append(row)
    rows = _drop_empty_cols(rows, fill_token="—")
    return _to_md_table(rows)


def _perf_detail_table(perf_rows: List[Dict[str, str]]) -> str:
    if not perf_rows:
        return ""
    header = ["exp_id", "scene", "backend", "mode",
              "conc", "bs", "xRT", "cps",
              "lat_p50", "lat_p95", "lat_p99", "rtf_mean"]
    rows = [header]
    for r in perf_rows:
        rows.append([
            r.get("exp_id", ""), r.get("scene", ""),
            r.get("backend", ""), r.get("mode", ""),
            r.get("concurrency") or "—",
            r.get("batch_size") or "—",
            _fmt(r.get("throughput_xrt", ""), 2),
            _fmt(r.get("throughput_cps", ""), 2),
            _fmt(r.get("latency_p50", ""), _PERF_NDIGITS),
            _fmt(r.get("latency_p95", ""), _PERF_NDIGITS),
            _fmt(r.get("latency_p99", ""), _PERF_NDIGITS),
            _fmt(r.get("rtf_mean", ""), _PERF_NDIGITS),
        ])
    return _to_md_table(rows)


def _best_perf_section(reg: List[Dict[str, str]]) -> str:
    scopes = _discover_perf_scopes(reg)
    if not scopes:
        return ""
    def _safe(r, k):
        try: return float(r.get(k) or "nan")
        except Exception: return float("nan")
    md = [["scope", "指标", "exp_id", "name", "值"]]
    for s in scopes:
        cand = [(r, _safe(r, f"xrt_{s}")) for r in reg]
        cand = [c for c in cand if c[1] == c[1]]
        if cand:
            best, val = max(cand, key=lambda x: x[1])
            md.append([s, "xRT max", best.get("exp_id", ""),
                       best.get("name", ""), f"{val:.2f}"])
        cand = [(r, _safe(r, f"latp95_{s}")) for r in reg]
        cand = [c for c in cand if c[1] == c[1]]
        if cand:
            best, val = min(cand, key=lambda x: x[1])
            md.append([s, "P95 min", best.get("exp_id", ""),
                       best.get("name", ""), f"{val:.3f}s"])
    if len(md) == 1:
        return ""
    return _to_md_table(md)


# ─── 主入口 ────────────────────────────────────────────────────────────

def main() -> None:
    reg = _read_csv(REGISTRY)
    pc = _read_csv(PER_CMD)
    pp = _read_csv(PER_PERF)
    agg = _aggregate_keywords(pc)
    ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")

    n_testsets = len({a["testset"] for a in agg})
    lines = [
        "# KWS 消融实验报告",
        "",
        f"_最后更新:{ts_now}  ·  共 {len(reg)} 个实验,"
        f"{n_testsets} 个 testset,{len(pc)} 条 per-keyword 行,"
        f"{len(pp)} 条 perf 记录_",
        "",
    ]

    if reg:
        lines += ["## 实验元数据", "", _experiment_meta_section(reg), ""]

    if agg:
        cov = _coverage_section(agg)
        if cov:
            lines += ["## 准确率 · 覆盖矩阵", "", cov, ""]

        lines += ["## 准确率 · 总览(长表)",
                  "",
                  "每行一个 (exp, testset, backend) 组合,无空格子;"
                  "数据全部从 `per_command.csv` 聚合,见 `keyword='all'` 行(若无,自动按 TP/FP/FN 加总)。",
                  "",
                  _long_summary_table(agg), ""]

        best = _best_section(agg)
        if best:
            lines += ["## 准确率 · 每个 testset 的最佳", "", best, ""]

        per_ts = _per_testset_section(agg)
        if per_ts:
            lines += ["## 准确率 · 按 testset 分组对比",
                      "",
                      "每张表自动剪掉了全空列。",
                      "",
                      per_ts, ""]

        sweep = _threshold_sweep_section(agg)
        if sweep:
            lines += ["## 准确率 · 阈值扫描(同实验/同 testset 多 tag)",
                      "",
                      sweep, ""]
    else:
        lines += ["## 准确率",
                  "",
                  "_尚无准确率数据。先 `python scripts/parse_decode.py runs/expXXX_...`_",
                  ""]

    # ── 性能章节 ──
    perf_sum = _perf_summary_table(reg)
    if perf_sum:
        lines += ["## 性能 · 总览", "", perf_sum, ""]
        best_perf = _best_perf_section(reg)
        if best_perf:
            lines += ["## 性能 · 最佳", "", best_perf, ""]
        perf_detail = _perf_detail_table(pp)
        if perf_detail:
            lines += ["## 性能 · 详细", "", perf_detail, ""]

    lines += [
        "## 怎么读这份报告",
        "",
        "### 准确率",
        "- **R / P / F1** = Recall / Precision / F1;**FA/h** = False alarms per negative hour(`eval.negative_hours[testset]` 提供)",
        "- **backend**:`pt`=icefall `decode.py`;`onnx`=sherpa-onnx;`onnx-tXX`=同模型扫阈值 0.XX",
        "- **tag**:同一个 backend 下用 `-t<value>` 区分超参扫描,会出现在专门的「阈值扫描」小节",
        "- 每个 testset 表里出现 `—` 是因为该组合没测过,不代表分数为 0",
        "- 完整 per-keyword 数据在 `per_command.csv`;宽表(每 testset/backend 一组列)仍保留在 `registry.csv`",
        "",
    ]
    if perf_sum:
        lines += [
            "### 性能",
            "- **xRT** = throughput / wall 音频实时倍率,越大越好",
            "- **cps** = calls per second",
            "- **lat_p95** = 端到端延迟 P95 (秒)",
            "- **rtf_mean** = 平均 Real-Time Factor(<1 表示比实时快)",
            "- 详细原始 JSON 见各实验的 `metrics/perf-*.json`,或 `per_perf.csv`",
            "- 打开 `scripts/report.ipynb` 可以看到交互图表",
            "",
        ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"已写入 {OUT}  ({len(lines)} 行)")


if __name__ == "__main__":
    main()
