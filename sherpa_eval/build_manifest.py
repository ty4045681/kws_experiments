#!/usr/bin/env python3
"""
从一个目录(可递归)生成 sherpa_onnx_kws_eval.py 能直接消费的 JSONL manifest。

三种模式(三选一):

  A) --transcript <file>
       转写文件,每行 "utt_id text"(icefall / kaldi 风格)
       脚本按文件名(去后缀)与 utt_id 匹配。

  B) --auto-pair
       每个 a.wav 旁边找同名 a.txt 作为转写。

  C) --fixed-text "<text>"      ← 新增
       忽略转写文件,所有音频共用同一段文本。
       适合"批量正样本"场景,例如这一批 wav 都是说同一个唤醒词。

公共行为:
  - --audio-dir 默认递归扫描子目录(--no-recursive 关闭)。
  - 输出的 text 中,空格会被替换为下划线 "_"(关闭用 --no-space-to-underscore)。
    这样写出来的 text 就是 sherpa-onnx KWS keywords.txt 里 phrase 那一侧
    常见的拼写形式。
  - 默认 text 转大写(--no-upper 关闭)。

用法:
  # 模式 A
  python build_manifest.py \\
      --audio-dir  /data/my_wavs/ \\
      --transcript /data/my_text \\
      --output     data/my_test.jsonl

  # 模式 B
  python build_manifest.py --audio-dir /data/my_wavs/ --auto-pair \\
      --output data/my_test.jsonl

  # 模式 C(新)
  python build_manifest.py \\
      --audio-dir   /data/wakeword_pos/ \\
      --fixed-text  "LIGHTS ON" \\
      --output      data/pos.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import wave
from pathlib import Path
from typing import Dict, List, Optional


def wav_duration(path: Path) -> Optional[float]:
    try:
        with wave.open(str(path)) as f:
            return f.getnframes() / float(f.getframerate())
    except (wave.Error, OSError):
        return None


def load_transcript(path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.strip():
                continue
            parts = line.split(None, 1)
            if len(parts) == 2:
                out[parts[0]] = parts[1]
    return out


def find_audio(audio_dir: Path, ext: str, recursive: bool) -> List[Path]:
    """递归(或不递归)收集音频文件,按相对路径稳定排序。"""
    pattern = f"**/*{ext}" if recursive else f"*{ext}"
    files = sorted(audio_dir.glob(pattern))
    # 过滤掉空文件/非文件
    return [p for p in files if p.is_file()]


def normalize_text(text: str, upper: bool, space_to_underscore: bool) -> str:
    text = text.strip()
    if upper:
        text = text.upper()
    if space_to_underscore:
        # 把任意连续空白(空格 / tab)折叠为单个下划线
        text = "_".join(text.split())
    return text


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--audio-dir", required=True, help="音频根目录")
    ap.add_argument("--ext", default=".wav", help="音频扩展名,默认 .wav")
    ap.add_argument("--recursive", dest="recursive", action="store_true", default=True,
                    help="递归进入子目录(默认开)")
    ap.add_argument("--no-recursive", dest="recursive", action="store_false",
                    help="只扫描 --audio-dir 顶层,不进入子目录")

    # 三种转写来源(互斥)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--transcript", help="模式 A:utt_id<空格>text 每行一条")
    grp.add_argument("--auto-pair", action="store_true",
                     help="模式 B:每个 a.wav 旁边找同名 a.txt")
    grp.add_argument("--fixed-text", help="模式 C:所有音频共用这段固定文本")

    # 文本规范化
    ap.add_argument("--upper", dest="upper", action="store_true", default=True,
                    help="文本转大写(默认开)")
    ap.add_argument("--no-upper", dest="upper", action="store_false")
    ap.add_argument("--space-to-underscore", dest="space_to_underscore",
                    action="store_true", default=True,
                    help="把文本里的空格替换成下划线(默认开)")
    ap.add_argument("--no-space-to-underscore", dest="space_to_underscore",
                    action="store_false")

    ap.add_argument("--output", required=True, help="输出 .jsonl 路径")
    ap.add_argument("--include-empty-text", action="store_true",
                    help="(模式 A / B)保留没有转写的音频,text 留空")
    ap.add_argument("--id-mode", choices=["stem", "relpath"], default="relpath",
                    help="utt id:'stem'=文件名去后缀;'relpath'=相对 audio-dir 的路径去后缀(默认,递归时避免重名冲突)")

    args = ap.parse_args()

    audio_dir = Path(args.audio_dir).resolve()
    if not audio_dir.exists():
        sys.exit(f"--audio-dir 不存在:{audio_dir}")
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    audios = find_audio(audio_dir, args.ext, args.recursive)
    if not audios:
        sys.exit(f"在 {audio_dir} 下没找到 *{args.ext}(recursive={args.recursive})")

    # 准备转写源
    texts: Dict[str, str] = {}
    if args.transcript:
        texts = load_transcript(Path(args.transcript))

    fixed_text_normalized: Optional[str] = None
    if args.fixed_text is not None:
        fixed_text_normalized = normalize_text(
            args.fixed_text, args.upper, args.space_to_underscore
        )
        print(f"[info] 固定文本(规范化后): {fixed_text_normalized!r}")

    n_written = 0
    n_skipped = 0
    seen_ids: Dict[str, Path] = {}

    with out_path.open("w", encoding="utf-8") as fo:
        for wav in audios:
            # utt_id
            if args.id_mode == "stem":
                uid = wav.stem
            else:
                rel = wav.relative_to(audio_dir).with_suffix("")
                uid = str(rel).replace("/", "__").replace("\\", "__")

            if uid in seen_ids:
                sys.stderr.write(
                    f"[warn] utt_id 冲突: {uid}  ({seen_ids[uid]} vs {wav});"
                    f"建议 --id-mode relpath\n"
                )
            seen_ids[uid] = wav

            # 取转写
            if args.fixed_text is not None:
                text_norm = fixed_text_normalized or ""
            elif args.auto_pair:
                t_path = wav.with_suffix(".txt")
                raw = t_path.read_text(encoding="utf-8").strip() if t_path.exists() else ""
                text_norm = normalize_text(raw, args.upper, args.space_to_underscore) if raw else ""
            else:
                # 模式 A:匹配先用 stem(默认 transcript 是这样写的)
                raw = texts.get(wav.stem, "")
                text_norm = normalize_text(raw, args.upper, args.space_to_underscore) if raw else ""

            if not text_norm and not args.include_empty_text:
                n_skipped += 1
                continue

            fo.write(json.dumps({
                "id": uid,
                "audio": str(wav.resolve()),
                "text": text_norm,
                "duration": wav_duration(wav),
            }, ensure_ascii=False) + "\n")
            n_written += 1

    print(f"[done] {out_path}  written={n_written}  skipped(no text)={n_skipped}  "
          f"scanned={len(audios)}  recursive={args.recursive}")


if __name__ == "__main__":
    main()
