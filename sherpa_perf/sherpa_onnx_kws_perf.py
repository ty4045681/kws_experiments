#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
sherpa-onnx KWS 推理性能测试。

测三件事:
  1. 时延       —— 端到端处理一条音频的耗时分布(P50/P90/P95/P99)
  2. RTF        —— 处理耗时 / 音频时长 (Real-Time Factor),越小越好
  3. 吞吐 / 并发 —— 一段时间内能稳定处理多少路、多少秒音频

输出 JSON(单一文件,所有场景累加)。文件名约定:
    perf-<scene>-<backend>[-tag].json
便于被上层 scripts/parse_perf.py 收纳进实验的 metrics.json。

──────────────────────────────────────────────────────────────────
三种测试模式
──────────────────────────────────────────────────────────────────
--mode single
    单线程顺序处理 manifest 里所有(或 --limit 条)音频,统计每条端到端
    延迟和 RTF。最贴近"测一条录音要多久"。

--mode concurrent
    起 --concurrency N 个工作线程,每个线程拥有独立 KeywordSpotter stream,
    从共享队列里取音频按"伪实时"或"全速"喂入。
    --pacing realtime   按 chunk 大小 sleep,模拟 N 路同时讲话的真实占用
    --pacing full       不 sleep,跑满 CPU,看吞吐上限
    用来回答:"我这台机器能稳定承载多少路并发?"

--mode batch
    用 sherpa-onnx 的 decode_streams([s1,...,sB]) 批量解码 API,
    --batch-size B 一次喂 B 条流,适合服务端 batch inference 场景。
    输出 batch RTF / 每条等效延迟。

公用计时:
    端到端耗时 = 从首次 accept_waveform 起,到最后 input_finished + 解空
                 之间的 wall-clock。
    RTF       = end_to_end_seconds / audio_seconds

──────────────────────────────────────────────────────────────────
manifest 格式(与 sherpa_eval 完全相同)
──────────────────────────────────────────────────────────────────
每行 JSON:{"audio": "/abs/path.wav", "text": "anything (perf 不用)"}
所以可以直接复用 sherpa_eval/build_manifest.py 的产物。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import statistics
import sys
import threading
import time
import wave
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

import numpy as np

# sherpa_onnx 是运行依赖;这里延迟到实际创建 spotter 时再导入，
# 这样 `--help` 和静态分析不会被拦住。
sherpa_onnx = None  # 占位，真正导入在 _ensure_sherpa() 里


def _ensure_sherpa():
    global sherpa_onnx
    if sherpa_onnx is not None:
        return sherpa_onnx
    try:
        import sherpa_onnx as _so  # type: ignore
    except ImportError:
        print("[error] 需要先安装 sherpa-onnx (pip install sherpa-onnx)", file=sys.stderr)
        raise
    sherpa_onnx = _so
    return sherpa_onnx


# ─── 输入读取 ───────────────────────────────────────────────────────────

@dataclass
class Sample:
    audio: str
    duration: float = 0.0  # 秒


def read_wave(path: str):
    with wave.open(path, "rb") as f:
        assert f.getnchannels() == 1, f"{path} 必须是单声道"
        assert f.getsampwidth() == 2, f"{path} 必须是 16-bit PCM"
        sr = f.getframerate()
        n = f.getnframes()
        raw = f.readframes(n)
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    return audio, sr


def load_manifest(path: str, limit: Optional[int]) -> List[Sample]:
    out: List[Sample] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            audio = obj.get("audio") or obj.get("path") or obj.get("wav")
            if not audio:
                continue
            try:
                with wave.open(audio, "rb") as w:
                    dur = w.getnframes() / float(w.getframerate())
            except (wave.Error, OSError):
                dur = 0.0
            out.append(Sample(audio=audio, duration=dur))
            if limit and len(out) >= limit:
                break
    return out


# ─── KeywordSpotter 构造 ────────────────────────────────────────────────

def build_spotter(args: argparse.Namespace) -> "sherpa_onnx.KeywordSpotter":
    _ensure_sherpa()
    kwargs = dict(
        tokens=args.tokens,
        encoder=args.encoder,
        decoder=args.decoder,
        joiner=args.joiner,
        keywords_file=args.keywords_file,
        num_threads=args.num_threads,
        provider=args.provider,
    )
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


# ─── 通用:跑一条音频(用于 single / concurrent),返回耗时秒 ─────────────

_TAIL_SECONDS = 0.66
_tail_cache: dict = {}


def _tail_padding(sr: int) -> np.ndarray:
    """每个采样率缓存一个尾部静音数组,避免热路径里 np.zeros 分配。"""
    a = _tail_cache.get(sr)
    if a is None:
        a = np.zeros(int(_TAIL_SECONDS * sr), dtype=np.float32)
        _tail_cache[sr] = a
    return a


def decode_one_blocking(kws, audio: np.ndarray, sr: int, chunk: int,
                        pacing: str = "full") -> float:
    """处理一条音频并返回端到端 wall-clock 秒数。

    pacing='realtime' 会按 chunk 长度 sleep,模拟真实流式输入;
    pacing='full'     不 sleep,直接全速喂,measure 纯计算开销。

    `chunk` 已是采样点数;调用方必须预先算好(避免热路径里 int() 与乘法)。
    """
    tail = _tail_padding(sr)
    s = kws.create_stream()
    t0 = time.perf_counter()
    for start in range(0, len(audio), chunk):
        seg = audio[start:start + chunk]
        s.accept_waveform(sr, seg)
        while kws.is_ready(s):
            kws.decode_stream(s)
            r = kws.get_result(s)
            if r:
                kws.reset_stream(s)
        if pacing == "realtime":
            time.sleep(len(seg) / sr)
    s.accept_waveform(sr, tail)
    s.input_finished()
    while kws.is_ready(s):
        kws.decode_stream(s)
        r = kws.get_result(s)
        if r:
            kws.reset_stream(s)
    return time.perf_counter() - t0


# ─── 三种模式 ──────────────────────────────────────────────────────────

@dataclass
class PerCallRecord:
    audio_seconds: float
    wall_seconds: float
    rtf: float


def _preload_pool(samples: List[Sample]) -> List[tuple]:
    """把所有 wav 一次性解码到内存。返回 [(audio, sr, dur), ...]。"""
    pool: List[tuple] = []
    for s in samples:
        try:
            audio, sr = read_wave(s.audio)
        except (wave.Error, OSError, ValueError) as e:
            print(f"[warn] 跳过 {s.audio}: {e}", file=sys.stderr)
            continue
        pool.append((audio, sr, len(audio) / sr))
    return pool


def run_single(kws, samples: List[Sample], chunk_seconds: float,
               warmup: int) -> dict:
    """单线程顺序跑。"""
    pool = _preload_pool(samples)
    if not pool:
        raise RuntimeError("manifest 中没有可用音频")
    sr0 = pool[0][1]
    chunk = max(1, int(chunk_seconds * sr0))

    for audio, sr, _ in pool[:warmup]:
        decode_one_blocking(kws, audio, sr, chunk if sr == sr0 else max(1, int(chunk_seconds * sr)),
                            pacing="full")

    recs: List[PerCallRecord] = []
    audio_total = 0.0
    t_all = time.perf_counter()
    for audio, sr, dur in pool:
        c = chunk if sr == sr0 else max(1, int(chunk_seconds * sr))
        wall = decode_one_blocking(kws, audio, sr, c, pacing="full")
        audio_total += dur
        recs.append(PerCallRecord(audio_seconds=dur, wall_seconds=wall,
                                  rtf=wall / dur if dur > 0 else 0.0))
    elapsed = time.perf_counter() - t_all
    return _summarize_single(recs, elapsed, audio_total)


def _latency_stats(walls: List[float], *, with_p90: bool = True,
                   with_p99: bool = True, with_max: bool = True) -> dict:
    """从一组耗时(秒)统计 mean/p50/p90/p95/p99/max。空列表返回 0.0。"""
    if not walls:
        d = {"mean": 0.0, "p50": 0.0, "p95": 0.0}
        if with_p90:
            d["p90"] = 0.0
        if with_p99:
            d["p99"] = 0.0
        if with_max:
            d["max"] = 0.0
        return d
    arr = np.asarray(walls, dtype=np.float64)
    d = {
        "mean": round(float(arr.mean()), 4),
        "p50":  round(float(np.percentile(arr, 50)), 4),
        "p95":  round(float(np.percentile(arr, 95)), 4),
    }
    if with_p90:
        d["p90"] = round(float(np.percentile(arr, 90)), 4)
    if with_p99:
        d["p99"] = round(float(np.percentile(arr, 99)), 4)
    if with_max:
        d["max"] = round(float(arr.max()), 4)
    return d


def _summarize_single(recs: List[PerCallRecord], elapsed: float,
                      audio_total: float) -> dict:
    walls = [r.wall_seconds for r in recs]
    rtfs = [r.rtf for r in recs]
    return {
        "n_samples": len(recs),
        "elapsed_seconds": round(elapsed, 4),
        "audio_seconds_total": round(audio_total, 4),
        "throughput_audio_per_wall": round(audio_total / elapsed, 4) if elapsed > 0 else 0.0,
        "latency_seconds": _latency_stats(walls),
        "rtf": _latency_stats(rtfs, with_p90=False, with_p99=False),
    }


def run_concurrent(kws_factory, samples: List[Sample], chunk_seconds: float,
                   concurrency: int, duration_seconds: float,
                   pacing: str, warmup: int) -> dict:
    """N 路并发。每个线程独立一个 KeywordSpotter,共享音频池循环喂。
    跑满 duration_seconds 后停。
    """
    pool = _preload_pool(samples)
    if not pool:
        raise RuntimeError("manifest 中没有可用音频")
    sr0 = pool[0][1]
    chunk = max(1, int(chunk_seconds * sr0))

    stop_flag = threading.Event()
    per_thread_recs: List[List[PerCallRecord]] = [[] for _ in range(concurrency)]

    def worker(tid: int):
        kws = kws_factory()
        for i in range(warmup):
            audio, sr, _ = pool[i % len(pool)]
            c = chunk if sr == sr0 else max(1, int(chunk_seconds * sr))
            decode_one_blocking(kws, audio, sr, c, pacing="full")
        i = 0
        while not stop_flag.is_set():
            audio, sr, dur = pool[i % len(pool)]
            i += 1
            c = chunk if sr == sr0 else max(1, int(chunk_seconds * sr))
            wall = decode_one_blocking(kws, audio, sr, c, pacing=pacing)
            per_thread_recs[tid].append(
                PerCallRecord(audio_seconds=dur, wall_seconds=wall,
                              rtf=wall / dur if dur > 0 else 0.0)
            )

    threads = [threading.Thread(target=worker, args=(i,), daemon=True)
               for i in range(concurrency)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    time.sleep(duration_seconds)
    stop_flag.set()
    for t in threads:
        t.join(timeout=60)
    elapsed = time.perf_counter() - t0

    all_recs = [r for lst in per_thread_recs for r in lst]
    audio_total = sum(r.audio_seconds for r in all_recs)
    walls = [r.wall_seconds for r in all_recs]
    rtfs = [r.rtf for r in all_recs]
    return {
        "concurrency": concurrency,
        "pacing": pacing,
        "duration_seconds": round(elapsed, 4),
        "n_calls_total": len(all_recs),
        "n_calls_per_thread_mean": round(statistics.fmean(
            [len(lst) for lst in per_thread_recs]), 2) if per_thread_recs else 0.0,
        "audio_seconds_total": round(audio_total, 4),
        "throughput_audio_per_wall": round(audio_total / elapsed, 4) if elapsed > 0 else 0.0,
        "throughput_calls_per_sec": round(len(all_recs) / elapsed, 4) if elapsed > 0 else 0.0,
        "latency_seconds": _latency_stats(walls),
        "rtf_per_stream": _latency_stats(rtfs, with_p90=False, with_p99=False, with_max=False),
    }


def run_batch(kws, samples: List[Sample], chunk_seconds: float,
              batch_size: int, n_batches: int, warmup: int) -> dict:
    """用 decode_streams([...]) 批量解码 API。

    每个 batch:同时 create_stream batch_size 条 -> 全量喂完 -> 反复
    decode_streams([active streams]) 直到所有 stream is_ready=False。
    """
    raw_pool = _preload_pool(samples)
    if not raw_pool:
        raise RuntimeError("manifest 中没有可用音频")
    pool = [(a, sr) for a, sr, _ in raw_pool]
    if len(pool) < batch_size:
        reps = (batch_size // len(pool)) + 1
        pool = (pool * reps)[:batch_size]
    sr0 = pool[0][1]
    chunk = max(1, int(chunk_seconds * sr0))
    tail = _tail_padding(sr0)

    def _run_one_batch() -> tuple:
        streams = [kws.create_stream() for _ in range(batch_size)]
        audio_total = 0.0
        t0 = time.perf_counter()
        for i, s in enumerate(streams):
            audio, sr = pool[i % len(pool)]
            audio_total += len(audio) / sr
            c = chunk if sr == sr0 else max(1, int(chunk_seconds * sr))
            for start in range(0, len(audio), c):
                s.accept_waveform(sr, audio[start:start + c])
            s.accept_waveform(sr, tail if sr == sr0 else _tail_padding(sr))
            s.input_finished()
        while True:
            ready = [s for s in streams if kws.is_ready(s)]
            if not ready:
                break
            kws.decode_streams(ready)
            for s in ready:
                r = kws.get_result(s)
                if r:
                    kws.reset_stream(s)
        wall = time.perf_counter() - t0
        return wall, audio_total

    for _ in range(max(0, warmup)):
        _run_one_batch()

    batch_walls: List[float] = []
    batch_audio: List[float] = []
    t_all = time.perf_counter()
    for _ in range(n_batches):
        wall, ad = _run_one_batch()
        batch_walls.append(wall)
        batch_audio.append(ad)
    total_elapsed = time.perf_counter() - t_all

    total_calls = n_batches * batch_size
    total_audio = sum(batch_audio)
    per_call_latency = [w / batch_size for w in batch_walls]
    return {
        "batch_size": batch_size,
        "n_batches": n_batches,
        "n_calls_total": total_calls,
        "elapsed_seconds": round(total_elapsed, 4),
        "audio_seconds_total": round(total_audio, 4),
        "throughput_audio_per_wall": round(total_audio / total_elapsed, 4) if total_elapsed > 0 else 0.0,
        "throughput_calls_per_sec": round(total_calls / total_elapsed, 4) if total_elapsed > 0 else 0.0,
        "batch_wall_seconds": _latency_stats(batch_walls, with_p90=False, with_p99=False),
        "per_call_latency_seconds": _latency_stats(per_call_latency, with_p90=False,
                                                    with_p99=False, with_max=False),
        "batch_rtf": {
            "mean": round(statistics.fmean(
                w / a for w, a in zip(batch_walls, batch_audio) if a > 0
            ), 4) if batch_walls else 0.0,
        },
    }


# ─── 环境信息 ──────────────────────────────────────────────────────────

def env_info(args) -> dict:
    info = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "provider": args.provider,
        "num_threads": args.num_threads,
    }
    info["sherpa_onnx_version"] = getattr(sherpa_onnx, "__version__", "unknown") if sherpa_onnx else "unknown"
    try:
        with open("/proc/cpuinfo", "r") as f:
            for line in f:
                if "model name" in line:
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return info


# ─── 主入口 ────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    # 模型
    p.add_argument("--tokens", required=True)
    p.add_argument("--encoder", required=True)
    p.add_argument("--decoder", required=True)
    p.add_argument("--joiner", required=True)
    p.add_argument("--keywords-file", required=True)
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--provider", default="cpu")
    p.add_argument("--keywords-score", type=float, default=None)
    p.add_argument("--keywords-threshold", type=float, default=None)
    p.add_argument("--num-trailing-blanks", type=int, default=None)
    p.add_argument("--max-active-paths", type=int, default=None)
    # 数据
    p.add_argument("--manifest", required=True, help="同 sherpa_eval 的 jsonl")
    p.add_argument("--limit", type=int, default=None,
                   help="只取前 N 条(对 single 模式是处理量;对 concurrent/batch 是音频池大小)")
    p.add_argument("--chunk-seconds", type=float, default=0.5)
    # 输出
    p.add_argument("--scene", required=True,
                   help="场景名(用于文件名和报告分组),例: cpu4t / cpu1t / batch16")
    p.add_argument("--suffix", default="onnx",
                   help="backend 后缀(默认 onnx)")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--tag", default="",
                   help="可选 tag,追加在文件名末尾,例 -c4 -b16")
    # 模式
    p.add_argument("--mode", required=True,
                   choices=["single", "concurrent", "batch"])
    # single 模式参数(已在通用里)
    # concurrent 模式参数
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--duration-seconds", type=float, default=30.0,
                   help="concurrent 模式持续时间")
    p.add_argument("--pacing", default="full", choices=["full", "realtime"])
    # batch 模式参数
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--n-batches", type=int, default=20)
    # 通用
    p.add_argument("--warmup", type=int, default=2,
                   help="预热次数(不计入统计)")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    _ensure_sherpa()

    print(f"[info] 加载 manifest: {args.manifest}", flush=True)
    samples = load_manifest(args.manifest, args.limit)
    if not samples:
        print("[error] manifest 为空", file=sys.stderr)
        sys.exit(1)
    print(f"[info] 样本数 = {len(samples)},音频总时长 ≈ "
          f"{sum(s.duration for s in samples):.2f}s", flush=True)

    print(f"[info] 构建 KeywordSpotter (provider={args.provider}, "
          f"num_threads={args.num_threads}) ...", flush=True)

    if args.mode == "single":
        kws = build_spotter(args)
        result = run_single(kws, samples, args.chunk_seconds, args.warmup)
    elif args.mode == "concurrent":
        # 每个线程一个 spotter(更贴近真实多实例部署)
        def _factory():
            return build_spotter(args)
        result = run_concurrent(_factory, samples, args.chunk_seconds,
                                args.concurrency, args.duration_seconds,
                                args.pacing, args.warmup)
    else:  # batch
        kws = build_spotter(args)
        result = run_batch(kws, samples, args.chunk_seconds,
                           args.batch_size, args.n_batches, args.warmup)

    payload = {
        "mode": args.mode,
        "scene": args.scene,
        "backend": args.suffix,
        "tag": args.tag,
        "chunk_seconds": args.chunk_seconds,
        "manifest": args.manifest,
        "n_manifest_samples": len(samples),
        "env": env_info(args),
        "model": {
            "encoder": args.encoder,
            "keywords_threshold": args.keywords_threshold,
            "keywords_score": args.keywords_score,
        },
        "result": result,
    }

    tag_part = f"-{args.tag}" if args.tag else ""
    out_path = out_dir / f"perf-{args.scene}-{args.suffix}{tag_part}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[done] 写入: {out_path}")
    # 打印关键指标摘要
    print("\n=== summary ===")
    r = result
    if args.mode == "single":
        print(f"  throughput  : {r['throughput_audio_per_wall']:.2f}x realtime")
        print(f"  latency P50 : {r['latency_seconds']['p50']:.3f}s")
        print(f"  latency P95 : {r['latency_seconds']['p95']:.3f}s")
        print(f"  RTF mean    : {r['rtf']['mean']:.3f}")
    elif args.mode == "concurrent":
        print(f"  concurrency : {r['concurrency']} ({r['pacing']})")
        print(f"  calls/sec   : {r['throughput_calls_per_sec']:.2f}")
        print(f"  audio xRT   : {r['throughput_audio_per_wall']:.2f}")
        print(f"  latency P95 : {r['latency_seconds']['p95']:.3f}s")
    else:
        print(f"  batch_size  : {r['batch_size']}")
        print(f"  calls/sec   : {r['throughput_calls_per_sec']:.2f}")
        print(f"  batch wall  : mean={r['batch_wall_seconds']['mean']:.3f}s")


if __name__ == "__main__":
    main()
