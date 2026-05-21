#!/usr/bin/env python3
"""
用 sherpa-onnx 对你自己的音频数据集做 KWS 评测。

输出文件与 icefall `egs/gigaspeech/KWS/zipformer/decode.py` **完全同格式**:
    metric-{testset}-{suffix}.txt

可以直接被上层脚手架的 scripts/parse_decode.py 解析(把它命名为
metric-<testset>-onnx[-tag].txt 放进对应实验的 metrics/ 目录即可)。

──────────────────────────────────────────────────────────────────
数据集格式(任选其一,都支持你自己的音频)

1) JSONL manifest(推荐,字段灵活)
   每行一条:
     {"audio": "/abs/path/a.wav",  "text": "TURN ON THE LIGHTS",  "duration": 2.1}
     {"audio": "/abs/path/b.wav",  "text": "HELLO WORLD",         "duration": 1.4}

   - audio    必填,16-bit PCM 单声道 wav(任意采样率,会自动重采到 16k)
   - text     必填,**大写**的参考转写;若是负样本(不含任何关键词),保持真实
              转写即可,脚本会按"是否包含 keyword"自动判定 TP/FP/FN/TN
   - duration 可选,浮点秒。若提供,会汇总用于"FA / hour"换算

2) TSV manifest:两列或三列,tab 分隔
     /abs/path/a.wav<TAB>TURN ON THE LIGHTS<TAB>2.1
     /abs/path/b.wav<TAB>HELLO WORLD<TAB>1.4
   有无第三列(duration)都行。

3) 目录 + 转写文件(快速试用)
     --audio-dir /data/wavs/   --transcript /data/text
   transcript 每行: <utt_id> <text>
   audio-dir 下的文件名(去掉 .wav)需与 utt_id 一一对应。

──────────────────────────────────────────────────────────────────
关键词文件(沿用 sherpa-onnx 原生格式)

每行一个 keyword,形如:
    LIGHTS ON @LIGHTS ON
    HEAT UP @HEAT UP
    y ǎn y uán @演员            # 中文 BPE pieces 也可以

@ 后面是 phrase(用于匹配 ref_text)。本脚本会:
  - 把 keywords_file 直接传给 sherpa-onnx KeywordSpotter
  - 同时把 @ 后面的 phrase 提取出来,做 TP/FP/FN/TN 统计(与 decode.py 一致)

──────────────────────────────────────────────────────────────────
典型用法

  python sherpa_onnx_kws_eval.py \
      --tokens   model/tokens.txt \
      --encoder  model/encoder-epoch-12-avg-2-chunk-16-left-128.onnx \
      --decoder  model/decoder-epoch-12-avg-2-chunk-16-left-128.onnx \
      --joiner   model/joiner-epoch-12-avg-2-chunk-16-left-128.onnx \
      --keywords-file model/keywords.txt \
      --manifest data/small_test.jsonl \
      --testset small \
      --suffix  onnx \
      --output-dir results/exp001_baseline/metrics/ \
      --keywords-threshold 0.35 \
      --keywords-score     1.0 \
      --chunk-seconds      0.5

  → 产生:
      results/exp001_baseline/metrics/metric-small-onnx.txt   (同 decode.py 格式)
      results/exp001_baseline/metrics/triggers-small-onnx.jsonl  (逐条 trigger,debug 用)
      results/exp001_baseline/metrics/summary-small-onnx.json    (机读摘要)

  之后可直接走脚手架链路:
      python scripts/parse_decode.py runs/exp001_baseline
      python scripts/update_registry.py runs/exp001_baseline
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import sherpa_onnx  # type: ignore
except ImportError as e:
    sys.stderr.write(
        "未找到 sherpa_onnx,请先安装:\n"
        "    pip install sherpa-onnx\n"
    )
    raise


# ─── 数据结构(与 decode.py 完全对齐)──────────────────────────────────────

@dataclass
class KwMetric:
    TP: int = 0
    FP: int = 0
    FN: int = 0
    TN: int = 0
    TP_list: List[str] = field(default_factory=list)
    FP_list: List[str] = field(default_factory=list)
    FN_list: List[str] = field(default_factory=list)


# ─── 数据集 loader ────────────────────────────────────────────────────────

@dataclass
class Sample:
    audio: Path
    text: str            # 已大写
    duration: Optional[float] = None
    utt_id: Optional[str] = None


def load_manifest_jsonl(path: Path) -> List[Sample]:
    out: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audio = Path(obj["audio"])
            text = str(obj.get("text", "")).upper()
            dur = obj.get("duration")
            out.append(Sample(audio=audio, text=text,
                              duration=float(dur) if dur is not None else None,
                              utt_id=obj.get("id") or audio.stem))
    return out


def load_manifest_tsv(path: Path) -> List[Sample]:
    out: List[Sample] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                raise ValueError(f"TSV 行少于 2 列:{line!r}")
            audio = Path(parts[0])
            text = parts[1].upper()
            dur = float(parts[2]) if len(parts) >= 3 and parts[2].strip() else None
            out.append(Sample(audio=audio, text=text, duration=dur, utt_id=audio.stem))
    return out


def load_from_dir(audio_dir: Path, transcript: Path) -> List[Sample]:
    """音频目录 + transcript 文件:<utt_id> <text>"""
    texts: Dict[str, str] = {}
    with transcript.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(None, 1)
            if len(parts) == 2:
                texts[parts[0]] = parts[1].upper()
    out: List[Sample] = []
    for wav in sorted(audio_dir.glob("*.wav")):
        uid = wav.stem
        if uid not in texts:
            sys.stderr.write(f"[warn] transcript 缺 {uid},跳过\n")
            continue
        out.append(Sample(audio=wav, text=texts[uid], utt_id=uid))
    return out


def load_dataset(args: argparse.Namespace) -> List[Sample]:
    if args.manifest:
        p = Path(args.manifest)
        if p.suffix.lower() == ".jsonl":
            return load_manifest_jsonl(p)
        if p.suffix.lower() in (".tsv", ".txt"):
            return load_manifest_tsv(p)
        raise ValueError(f"未知 manifest 后缀:{p.suffix},请用 .jsonl / .tsv")
    if args.audio_dir and args.transcript:
        return load_from_dir(Path(args.audio_dir), Path(args.transcript))
    raise SystemExit("请提供 --manifest,或同时提供 --audio-dir 与 --transcript")


# ─── 音频读取 ────────────────────────────────────────────────────────────

def read_wave(path: Path) -> Tuple[np.ndarray, int]:
    with wave.open(str(path)) as f:
        if f.getnchannels() != 1:
            raise ValueError(f"{path} 不是单声道")
        if f.getsampwidth() != 2:
            raise ValueError(f"{path} 不是 16-bit PCM")
        n = f.getnframes()
        raw = f.readframes(n)
        sr = f.getframerate()
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, sr


# ─── keywords 文件解析(只为评测;sherpa-onnx 自己也会解析一遍)────────

def _norm_phrase(s: str, space_to_underscore: bool) -> str:
    s = s.strip().upper()
    if space_to_underscore:
        s = "_".join(s.split())
    return s


def parse_keywords_phrases(path: Path, space_to_underscore: bool = True) -> List[str]:
    """从 sherpa-onnx keywords 文件提取 phrase(@后面),全部 UPPER。
    若 space_to_underscore=True(默认),phrase 中的空格替换为下划线,
    与 build_manifest.py 默认产出的 text 一致。"""
    phrases: List[str] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "@" in line:
                _, phrase = line.split("@", 1)
                phrase = _norm_phrase(phrase, space_to_underscore)
                if phrase:
                    phrases.append(phrase)
    # 保留首次出现顺序、去重
    seen = set()
    uniq = []
    for p in phrases:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq


# ─── KeywordSpotter 包装 ─────────────────────────────────────────────────

def build_spotter(args: argparse.Namespace) -> "sherpa_onnx.KeywordSpotter":
    kwargs = dict(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        keywords_file=args.keywords_file,
        num_threads=args.num_threads,
        provider=args.provider,
    )
    # 不同版本 sherpa-onnx 字段名兼容
    optional = {
        "keywords_score": args.keywords_score,
        "keywords_threshold": args.keywords_threshold,
        "num_trailing_blanks": args.num_trailing_blanks,
        "max_active_paths": args.max_active_paths,
    }
    for k, v in optional.items():
        if v is not None:
            kwargs[k] = v
    return sherpa_onnx.KeywordSpotter(**kwargs)


# ─── 一条音频:推流 + 收集 trigger ────────────────────────────────────────

def decode_one(kws, sample: Sample, chunk_seconds: float,
               space_to_underscore: bool = True,
               target_sr: int = 16000) -> List[str]:
    audio, sr = read_wave(sample.audio)
    s = kws.create_stream()
    # sherpa-onnx 接受任意 sr,内部会重采;直接给它原 sr
    chunk = max(1, int(chunk_seconds * sr))
    triggers: List[str] = []
    for start in range(0, len(audio), chunk):
        s.accept_waveform(sr, audio[start:start + chunk])
        while kws.is_ready(s):
            kws.decode_stream(s)
            r = kws.get_result(s)
            if r:
                triggers.append(_norm_phrase(r, space_to_underscore))
                kws.reset_stream(s)
    # 尾部 padding(官方示例 0.66s),触发可能在最后帧
    tail = np.zeros(int(0.66 * sr), dtype=np.float32)
    s.accept_waveform(sr, tail)
    s.input_finished()
    while kws.is_ready(s):
        kws.decode_stream(s)
        r = kws.get_result(s)
        if r:
            triggers.append(_norm_phrase(r, space_to_underscore))
            kws.reset_stream(s)
    return triggers


# ─── 判定:与 decode.py 1:1 对齐 ────────────────────────────────────────

def judge(
    ref_text: str,
    hyp_set: List[str],
    keywords: List[str],
    metric: Dict[str, KwMetric],
    test_only_keywords: bool,
) -> Tuple[bool, bool, bool, bool, str]:
    """返回 (TP, FP, TN, FN, hyp_str),并就地更新 metric。"""
    ref_text = ref_text.upper()
    hyp_str = " | ".join(hyp_set)

    TP = FP = False
    for x in set(hyp_set):
        if x not in keywords:
            # sherpa-onnx 触发的短语未声明在 keywords 里:计为虚警
            metric.setdefault(x, KwMetric())
            FP = True
            metric[x].FP += 1
            metric[x].FP_list.append(f"({ref_text} -> {x})")
            continue
        if (test_only_keywords and x == ref_text) or (
            not test_only_keywords and x in ref_text
        ):
            TP = True
            metric[x].TP += 1
            metric[x].TP_list.append(f"({ref_text} -> {x})")
        if (test_only_keywords and x != ref_text) or (
            not test_only_keywords and x not in ref_text
        ):
            FP = True
            metric[x].FP += 1
            metric[x].FP_list.append(f"({ref_text} -> {x})")
    if TP: metric["all"].TP += 1
    if FP: metric["all"].FP += 1

    TN = True
    FN = False
    hyp_set_uniq = set(hyp_set)
    for x in keywords:
        if x not in ref_text and x not in hyp_set_uniq:
            metric[x].TN += 1
            continue
        TN = False
        if (test_only_keywords and x == ref_text) or (
            not test_only_keywords and x in ref_text
        ):
            fn = True
            for y in hyp_set_uniq:
                if (test_only_keywords and y == ref_text) or (
                    not test_only_keywords and y in ref_text
                ):
                    fn = False
                    break
            if fn:
                FN = True
                metric[x].FN += 1
                metric[x].FN_list.append(f"({ref_text} -> {hyp_str})")
    if TN: metric["all"].TN += 1
    if FN: metric["all"].FN += 1
    return TP, FP, TN, FN, hyp_str


# ─── 输出 metric-*.txt(decode.py 同款)──────────────────────────────────

def write_metric_file(out_path: Path, metric: Dict[str, KwMetric], dump_lists: bool = True) -> None:
    width = 10
    s_lines: List[str] = []
    # 与 decode.py 一致:按 (FP, FN) 倒序
    items = sorted(metric.items(), key=lambda kv: (kv[1].FP, kv[1].FN), reverse=True)
    for name, item in items:
        denom_acc = item.TP + item.TN + item.FP + item.FN
        acc = (item.TP + item.TN) / denom_acc if denom_acc else 0.0
        prec = 0.0 if (item.TP + item.FP) == 0 else item.TP / (item.TP + item.FP)
        rec = 0.0 if (item.TP + item.FN) == 0 else item.TP / (item.TP + item.FN)
        fpr = 0.0 if (item.FP + item.TN) == 0 else item.FP / (item.FP + item.TN)
        f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

        s = f"{name}:\n"
        s += f"\t{'TP':{width}}{'FP':{width}}{'FN':{width}}{'TN':{width}}\n"
        s += f"\t{str(item.TP):{width}}{str(item.FP):{width}}{str(item.FN):{width}}{str(item.TN):{width}}\n"
        s += f"\tAccuracy: {acc:.3f}\n"
        s += f"\tPrecision: {prec:.3f}\n"
        s += f"\tRecall(PPR): {rec:.3f}\n"
        s += f"\tFPR: {fpr:.3f}\n"
        s += f"\tF1: {f1:.3f}\n"
        if dump_lists and name != "all":
            s += f"\tTP list: {' # '.join(item.TP_list)}\n"
            s += f"\tFP list: {' # '.join(item.FP_list)}\n"
            s += f"\tFN list: {' # '.join(item.FN_list)}\n"
        s += "\n"
        s_lines.append(s)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(s_lines), encoding="utf-8")


# ─── 主流程 ────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # 模型
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--decoder", required=True)
    ap.add_argument("--joiner", required=True)
    ap.add_argument("--keywords-file", required=True)
    ap.add_argument("--provider", default="cpu")
    ap.add_argument("--num-threads", type=int, default=2)

    # 解码参数
    ap.add_argument("--keywords-score", type=float, default=None,
                    help="对应 sherpa-onnx 的 keywords_score (默认让其用内部默认)")
    ap.add_argument("--keywords-threshold", type=float, default=None,
                    help="对应 sherpa-onnx 的 keywords_threshold")
    ap.add_argument("--num-trailing-blanks", type=int, default=None)
    ap.add_argument("--max-active-paths", type=int, default=None)
    ap.add_argument("--chunk-seconds", type=float, default=0.5,
                    help="按多长一段往流里推(模拟实时)。默认 0.5s")

    # 数据集
    ap.add_argument("--manifest", help="JSONL 或 TSV 清单")
    ap.add_argument("--audio-dir", help="音频目录,需配合 --transcript")
    ap.add_argument("--transcript", help="<utt_id> <text> 每行一条")

    # 判定模式
    ap.add_argument("--test-only-keywords", action="store_true",
                    help="开启后只匹配整段 ref==keyword;否则匹配 'keyword in ref'(与 decode.py 默认一致)")

    # 输出
    ap.add_argument("--testset", default="small", help="写入文件名 metric-{testset}-{suffix}.txt")
    ap.add_argument("--suffix", default="onnx")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--no-debug-lists", action="store_true", help="不要在 metric-*.txt 里写 TP/FP/FN 明细列表")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 条(调试用)")
    ap.add_argument("--phrase-space-to-underscore", dest="phrase_space_to_underscore",
                    action="store_true", default=True,
                    help="把 keywords 文件里 @ 后面的 phrase、以及 sherpa-onnx 返回的 trigger 中的空格全部替换为下划线,以便和 build_manifest.py 默认产出的 text 一致(默认开)")
    ap.add_argument("--no-phrase-space-to-underscore", dest="phrase_space_to_underscore",
                    action="store_false")

    args = ap.parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    keywords = parse_keywords_phrases(
        Path(args.keywords_file), space_to_underscore=args.phrase_space_to_underscore,
    )
    if not keywords:
        sys.exit("未从 --keywords-file 解析出任何 phrase(@ 后内容),请检查格式")
    print(f"[info] keywords ({len(keywords)}): {keywords}")

    samples = load_dataset(args)
    if args.limit:
        samples = samples[: args.limit]
    if not samples:
        sys.exit("数据集为空")
    print(f"[info] 共 {len(samples)} 条音频")

    kws = build_spotter(args)

    metric: Dict[str, KwMetric] = OrderedDict()
    metric["all"] = KwMetric()
    for k in keywords:
        metric[k] = KwMetric()

    triggers_path = out_dir / f"triggers-{args.testset}-{args.suffix}.jsonl"
    total_audio_sec = 0.0
    total_neg_sec = 0.0  # 不含任何关键词的音频累计时长 → 用于 FA/hour
    t0 = time.time()

    with triggers_path.open("w", encoding="utf-8") as ft:
        for i, sp in enumerate(samples, 1):
            triggers = decode_one(
                kws, sp, chunk_seconds=args.chunk_seconds,
                space_to_underscore=args.phrase_space_to_underscore,
            )
            tp, fp, tn, fn, hyp_str = judge(
                sp.text, triggers, keywords, metric, args.test_only_keywords
            )
            if sp.duration is not None:
                total_audio_sec += sp.duration
                # 负样本 = ref 不包含任何 keyword
                if not any(k in sp.text for k in keywords):
                    total_neg_sec += sp.duration
            ft.write(json.dumps({
                "utt_id": sp.utt_id,
                "audio": str(sp.audio),
                "ref": sp.text,
                "triggers": triggers,
                "TP": tp, "FP": fp, "TN": tn, "FN": fn,
            }, ensure_ascii=False) + "\n")
            if i % 50 == 0 or i == len(samples):
                m = metric["all"]
                print(f"[{i}/{len(samples)}] TP={m.TP} FP={m.FP} FN={m.FN} TN={m.TN}")

    metric_file = out_dir / f"metric-{args.testset}-{args.suffix}.txt"
    write_metric_file(metric_file, metric, dump_lists=not args.no_debug_lists)

    # 机读摘要
    m_all = metric["all"]
    elapsed = time.time() - t0
    summary = {
        "testset": args.testset,
        "suffix": args.suffix,
        "n_samples": len(samples),
        "elapsed_sec": round(elapsed, 2),
        "total_audio_sec": round(total_audio_sec, 2),
        "total_negative_sec": round(total_neg_sec, 2),
        "rtf": round(elapsed / total_audio_sec, 4) if total_audio_sec else None,
        "all": {
            "TP": m_all.TP, "FP": m_all.FP, "FN": m_all.FN, "TN": m_all.TN,
            "recall": (m_all.TP / (m_all.TP + m_all.FN)) if (m_all.TP + m_all.FN) else 0.0,
            "fa_per_hour": (m_all.FP / (total_neg_sec / 3600)) if total_neg_sec > 0 else None,
        },
        "keywords": keywords,
        "args": {
            "keywords_score": args.keywords_score,
            "keywords_threshold": args.keywords_threshold,
            "chunk_seconds": args.chunk_seconds,
            "test_only_keywords": args.test_only_keywords,
        },
    }
    (out_dir / f"summary-{args.testset}-{args.suffix}.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[done] 写入:\n  {metric_file}\n  {triggers_path}\n"
          f"  {out_dir / f'summary-{args.testset}-{args.suffix}.json'}")
    print(f"[done] 总览  recall={summary['all']['recall']:.3f}  "
          f"FA/h={summary['all']['fa_per_hour']}  RTF={summary['rtf']}")


if __name__ == "__main__":
    main()
