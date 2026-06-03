#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
查看 runs/expNNN/config.yaml 可配置项。

常用:
  python scripts/config_help.py                 # 列出所有配置点
  python scripts/config_help.py eval.testsets   # 查看某个配置点
  python scripts/config_help.py perf.scenes     # 查看性能场景字段
  python scripts/config_help.py --markdown      # 输出完整 Markdown
  python scripts/config_help.py --write CONFIG_REFERENCE.md

设计目标:让 config.yaml 保持简洁,把完整字段说明集中在一个可查询/可生成文档的地方。
"""

from __future__ import annotations

import argparse
import sys
import textwrap
from pathlib import Path
from typing import Dict, List, Sequence

ROOT = Path(__file__).resolve().parent.parent

Field = Dict[str, str]
Section = Dict[str, object]


def F(name: str, typ: str, default: str, desc: str) -> Field:
    return {"name": name, "type": typ, "default": default, "desc": desc}


SECTIONS: List[Section] = [
    {
        "name": "meta",
        "title": "meta — 实验元信息",
        "desc": "由 new_experiment.py 自动填一部分；用于 registry.csv / REPORT.md 展示。",
        "fields": [
            F("exp_id", "str", "自动填，如 exp003", "实验编号；必须形如 expNNN。"),
            F("name", "str", "自动填", "实验短名；会显示在报告中。"),
            F("date", "str", "自动填", "实验创建日期。"),
            F("variable", "str", "可选", "本次消融变量名，如 lr / threshold / chunk_size。"),
            F("value", "str", "可选", "变量取值；建议保持字符串，方便比较。"),
            F("notes", "str", "可选", "自由备注，会显示在实验元数据表中。"),
            F("ckpt_path", "path", "可选", "checkpoint 路径，仅作记录。"),
        ],
        "example": """
meta:
  exp_id: exp003
  name: lights_on_t035
  variable: threshold
  value: "0.35"
  notes: "fixed-text 正样本 + car_noise 负样本"
""",
    },
    {
        "name": "model",
        "title": "model — sherpa-onnx 模型文件",
        "desc": "运行 eval / sweep / perf 时必填。相对路径按项目根目录解析；也支持 ${ROOT} / ${EXP_DIR}。",
        "fields": [
            F("tokens", "path", "必填", "tokens.txt。"),
            F("encoder", "path", "必填", "encoder ONNX 文件。"),
            F("decoder", "path", "必填", "decoder ONNX 文件。"),
            F("joiner", "path", "必填", "joiner ONNX 文件。"),
            F("keywords_file", "path", "必填", "sherpa-onnx keywords.txt；每行 '<tokens> @<phrase>'。"),
        ],
        "example": """
model:
  tokens: sherpa_eval/model/tokens.txt
  encoder: sherpa_eval/model/encoder.onnx
  decoder: sherpa_eval/model/decoder.onnx
  joiner: sherpa_eval/model/joiner.onnx
  keywords_file: sherpa_eval/model/keywords.txt
""",
    },
    {
        "name": "train",
        "title": "train — 训练超参记录",
        "desc": "仅用于记录和横向对照，不会被 icefall 训练脚本直接读取。部分字段会平铺到 registry.csv。",
        "fields": [
            F("stage", "str", "pretrain", "训练阶段，如 pretrain / finetune。"),
            F("base_model", "str", "可选", "基础模型名。"),
            F("subset", "str", "可选", "训练数据子集。"),
            F("bpe_size", "int", "可选", "BPE 大小。"),
            F("num_epochs", "int", "可选", "训练 epoch 数。"),
            F("base_lr", "float", "可选", "学习率。"),
            F("lr_epochs", "float", "可选", "icefall run.sh 对应字段。"),
            F("lr_batches", "int", "可选", "icefall run.sh 对应字段。"),
            F("max_duration", "float/int", "可选", "训练 max_duration。"),
            F("use_fp16", "bool", "可选", "是否使用 fp16。"),
            F("causal", "bool", "可选", "是否 causal。"),
            F("num_encoder_layers", "str", "可选", "结构记录，如 '1,1,1,1,1,1'。"),
            F("feedforward_dim", "str", "可选", "结构记录。"),
            F("encoder_dim", "str", "可选", "结构记录。"),
            F("encoder_unmasked_dim", "str", "可选", "结构记录。"),
            F("decoder_dim", "int", "可选", "结构记录。"),
            F("joiner_dim", "int", "可选", "结构记录。"),
        ],
        "example": """
train:
  stage: finetune
  base_model: zipformer-3.3M
  base_lr: 0.0005
  num_epochs: 12
  causal: true
""",
    },
    {
        "name": "decode",
        "title": "decode — 解码/评测参数记录",
        "desc": "用于记录 icefall decode / 评测口径；eval 中未填写时，run_from_config.py 会回退读取部分字段。",
        "fields": [
            F("epoch", "int", "可选", "decode 使用的 epoch。"),
            F("avg", "int", "可选", "checkpoint averaging 数。"),
            F("chunk_size", "int", "可选", "decode chunk size。"),
            F("left_context_frames", "int", "可选", "decode left context。"),
            F("keywords_score", "float", "可选", "eval.keywords_score 为空时的回退值。"),
            F("keywords_threshold", "float", "可选", "eval.keywords_threshold 为空时的回退值。"),
            F("max_duration", "float/int", "可选", "decode max_duration。"),
        ],
        "example": """
decode:
  epoch: 12
  avg: 2
  chunk_size: 16
  left_context_frames: 64
  keywords_score: 1.0
  keywords_threshold: 0.35
""",
    },
    {
        "name": "onnx",
        "title": "onnx — 部署/导出记录",
        "desc": "仅作记录，便于在 REPORT.md / registry.csv 中和性能结果对齐。",
        "fields": [
            F("exported", "bool", "false", "是否已导出 ONNX。"),
            F("quant", "str", "fp32", "量化类型，如 fp32 / fp16 / int8。"),
            F("chunk_size", "int", "可选", "导出时 chunk size。"),
            F("left_context_frames", "int", "可选", "导出时 left context。"),
        ],
        "example": """
onnx:
  exported: true
  quant: fp32
  chunk_size: 16
  left_context_frames: 128
""",
    },
    {
        "name": "eval",
        "title": "eval — 准确率评测总配置",
        "desc": "run_from_config.py 的 manifest / eval / sweep stage 会读取这一段。",
        "fields": [
            F("manifest_dir", "path", "sherpa_eval/data", "build_manifest.py 输出目录；REPORT.md 也默认从这里读 duration。"),
            F("suffix", "str", "onnx", "metric 文件 backend 后缀，如 onnx / onnx-int8。"),
            F("provider", "str", "cpu", "传给 sherpa-onnx，如 cpu。"),
            F("num_threads", "int", "2", "KeywordSpotter num_threads。"),
            F("chunk_seconds", "float", "0.5", "流式喂入 chunk 秒数。"),
            F("keywords_threshold", "float", "decode.keywords_threshold", "KWS 阈值；越高越保守。"),
            F("keywords_score", "float", "decode.keywords_score", "关键词加成分。"),
            F("num_trailing_blanks", "int", "可选", "传给 sherpa-onnx KeywordSpotter。"),
            F("max_active_paths", "int", "可选", "传给 sherpa-onnx KeywordSpotter。"),
            F("limit", "int/null", "null", "调试时只跑前 N 条。"),
            F("debug_lists", "bool", "true", "false 时 metric 文件不写 TP/FP/FN 明细列表。"),
            F("test_only_keywords", "bool", "false", "true 时严格 ref == keyword；false 时 keyword in ref。"),
            F("phrase_space_to_underscore", "bool", "true", "keywords phrase 和 trigger 空格转下划线。"),
            F("thresholds", "list[str/float]", "[]", "sweep stage 使用；生成 suffix=onnx-tXX。"),
            F("testsets", "list[dict]", "[]", "测试集列表；详见 eval.testsets。"),
            F("negative_hours", "dict", "small/large 默认", "testset -> 负样本小时数，用于 FP 换算 FA/hour。"),
            F("extra_args", "str/list", "可选", "追加给 sherpa_onnx_kws_eval.py 的额外 CLI 参数。字符串按 shell 规则拆分。"),
        ],
        "example": """
eval:
  manifest_dir: sherpa_eval/data
  suffix: onnx
  provider: cpu
  num_threads: 2
  chunk_seconds: 0.5
  keywords_threshold: 0.35
  thresholds: ["0.20", "0.25", "0.30", "0.35"]
  negative_hours:
    car_noise: 2.1
""",
    },
    {
        "name": "eval.testsets",
        "title": "eval.testsets[] — 准确率测试集配置",
        "desc": "每个元素定义一个 testset。manifest stage 生成 JSONL；eval/sweep stage 消费 JSONL。",
        "fields": [
            F("name", "str", "必填", "testset 名；会出现在 metric-<name>-onnx.txt 和报告中。"),
            F("manifest", "path", "可选", "已有 JSONL/TSV manifest；填了它可省略 mode/audio_dir。"),
            F("mode", "transcript | auto-pair | fixed-text", "manifest 为空时必填", "决定 build_manifest.py 的转写来源。"),
            F("audio_dir", "path", "mode 需要", "音频根目录。"),
            F("transcript", "path", "mode=transcript", "每行 '<utt_id> text' 的转写文件。"),
            F("text / fixed_text", "str", "mode=fixed-text", "所有音频共用文本；适合批量正样本。"),
            F("ext", "str", ".wav", "音频扩展名。"),
            F("recursive", "bool", "true", "是否递归扫描 audio_dir。false 会加 --no-recursive。"),
            F("upper", "bool", "true", "是否把 text 转大写。false 会加 --no-upper。"),
            F("space_to_underscore", "bool", "true", "是否把空格转下划线。false 会加 --no-space-to-underscore。"),
            F("include_empty_text", "bool", "false", "是否保留无转写音频。"),
            F("id_mode", "stem | relpath", "relpath", "utt_id 生成方式。递归目录建议 relpath。"),
            F("extra_eval_args", "str/list", "可选", "只追加给这个 testset 的 eval CLI 参数。"),
        ],
        "example": """
eval:
  testsets:
    - name: wakeword_pos
      mode: fixed-text
      audio_dir: /data/wakeword_pos
      text: "lights on"

    - name: car_noise
      mode: transcript
      audio_dir: /data/car_noise
      transcript: /data/car_noise.text

    - name: kitchen
      mode: auto-pair
      audio_dir: /data/kitchen

    - name: prebuilt
      manifest: sherpa_eval/data/prebuilt.jsonl
""",
    },
    {
        "name": "perf",
        "title": "perf — 性能测试总配置",
        "desc": "run_from_config.py 的 perf stage 会读取这一段。scene 内字段会覆盖 perf 总配置。",
        "fields": [
            F("hardware", "str", "可选", "硬件描述，仅作记录。"),
            F("notes", "str", "可选", "性能测试备注，仅作记录。"),
            F("suffix", "str", "eval.suffix/onnx", "perf 文件 backend 后缀。"),
            F("provider", "str", "eval.provider/cpu", "传给 sherpa-onnx。"),
            F("num_threads", "int", "eval.num_threads/1", "默认 spotter num_threads；scene 可覆盖。"),
            F("chunk_seconds", "float", "eval.chunk_seconds/0.5", "默认 chunk 秒数；scene 可覆盖。"),
            F("warmup", "int", "2", "预热次数。"),
            F("pacing", "full | realtime", "full", "concurrent/batch_streaming/cpu_sweep 默认 pacing。"),
            F("duration_seconds", "float", "30", "持续压测时长；scene 可覆盖。"),
            F("testset", "str", "可选", "性能测试默认音频池，引用 eval.testsets[].name。"),
            F("manifest", "path", "可选", "性能测试默认 manifest；优先级高于 testset。"),
            F("limit", "int", "可选", "默认只取 manifest 前 N 条。"),
            F("keywords_threshold", "float", "eval/decode 回退", "性能测试构建 spotter 的阈值。"),
            F("keywords_score", "float", "eval/decode 回退", "性能测试构建 spotter 的关键词加成。"),
            F("num_trailing_blanks", "int", "可选", "传给 sherpa-onnx。"),
            F("max_active_paths", "int", "可选", "传给 sherpa-onnx。"),
            F("scenes", "list[dict]", "[]", "性能场景列表；详见 perf.scenes。"),
            F("extra_args", "str/list", "可选", "追加给所有 perf CLI 的额外参数。"),
        ],
        "example": """
perf:
  testset: wakeword_pos
  suffix: onnx
  provider: cpu
  scenes:
    - scene: single_cpu1t
      mode: single
      num_threads: 1
""",
    },
    {
        "name": "perf.scenes",
        "title": "perf.scenes[] — 性能测试场景配置",
        "desc": "每个元素会生成一份 perf-<scene>-<suffix>[-tag].json。字段按 mode 生效。",
        "fields": [
            F("scene", "str", "必填", "场景名；用于文件名和报告列名。"),
            F("mode", "single | concurrent | batch | batch_streaming | cpu_sweep", "必填", "性能测试模式。"),
            F("testset", "str", "可选", "覆盖 perf.testset，引用 eval.testsets[].name。"),
            F("manifest", "path", "可选", "覆盖 perf.manifest。"),
            F("suffix", "str", "继承 perf.suffix", "backend 后缀。"),
            F("tag", "str", "可选", "追加到文件名，如 int8 / c8。"),
            F("provider", "str", "继承 perf.provider", "传给 sherpa-onnx。"),
            F("num_threads", "int", "继承 perf.num_threads", "spotter num_threads。"),
            F("chunk_seconds", "float", "继承 perf.chunk_seconds", "chunk 秒数。"),
            F("limit", "int", "继承 perf.limit", "只取前 N 条音频。"),
            F("warmup", "int", "继承 perf.warmup", "预热次数。"),
            F("keywords_threshold", "float", "继承 perf/eval/decode", "构建 spotter 的阈值。"),
            F("keywords_score", "float", "继承 perf/eval/decode", "构建 spotter 的关键词加成。"),
            F("extra_args", "str/list", "可选", "只追加给该 scene 的额外 CLI 参数。"),
            F("concurrency", "int", "mode=concurrent/batch_streaming", "并发路数；batch_streaming 中也等价于活跃 stream 数。"),
            F("duration_seconds", "float", "mode=concurrent/batch_streaming/cpu_sweep", "持续压测秒数。"),
            F("pacing", "full | realtime", "mode=concurrent/batch_streaming/cpu_sweep", "full 跑满 CPU；realtime 模拟真实麦克风。"),
            F("batch_size", "int", "mode=batch", "offline batch 大小。batch_streaming 可用 concurrency。"),
            F("n_batches", "int", "mode=batch", "offline batch 轮数。"),
            F("inner_mode", "concurrent | batch_streaming", "mode=cpu_sweep", "cpu_sweep 内层模式。"),
            F("concurrency_list", "str", "mode=cpu_sweep", "逗号分隔并发点，如 '1,2,4,8,16'。"),
            F("target_cpu", "float", "mode=cpu_sweep", "CPU 预算阈值。"),
            F("cpu_budget_mode", "per_core | total", "mode=cpu_sweep", "CPU% 口径。"),
            F("cpu_affinity", "str", "mode=cpu_sweep/Linux", "绑核，如 '0' / '0-3' / '0,2,4'。"),
        ],
        "example": """
perf:
  testset: wakeword_pos
  scenes:
    - scene: single_cpu1t
      mode: single
      num_threads: 1
      limit: 100

    - scene: concurrent_c8
      mode: concurrent
      concurrency: 8
      duration_seconds: 30
      pacing: realtime

    - scene: batch_b8
      mode: batch
      batch_size: 8
      n_batches: 20

    - scene: batch_streaming_b8
      mode: batch_streaming
      concurrency: 8
      duration_seconds: 30
      pacing: realtime

    - scene: cpu_sweep_c
      mode: cpu_sweep
      inner_mode: concurrent
      concurrency_list: "1,2,4,8,16"
      target_cpu: 70
      cpu_budget_mode: per_core
""",
    },
]

ALIASES = {
    "testsets": "eval.testsets",
    "eval.testset": "eval.testsets",
    "perf.scene": "perf.scenes",
    "scenes": "perf.scenes",
    "scene": "perf.scenes",
}


def section_names() -> List[str]:
    return [str(s["name"]) for s in SECTIONS]


def find_section(name: str) -> Section:
    key = ALIASES.get(name, name)
    for s in SECTIONS:
        if s["name"] == key:
            return s
    choices = ", ".join(section_names())
    raise SystemExit(f"未知配置点: {name!r}\n可选配置点: {choices}")


def md_table(fields: Sequence[Field]) -> str:
    def cell(v: str) -> str:
        # Markdown 表格单元格中裸 `|` 会被当成列分隔符；
        # enum 类型常写成 "a | b | c"，这里统一转义，避免生成坏表格。
        return str(v).replace("\n", "<br>").replace("|", r"\|")

    lines = [
        "| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |",
        "|---|---|---|---|",
    ]
    for f in fields:
        lines.append(
            f"| `{cell(f['name'])}` | {cell(f['type'])} | {cell(f['default'])} | {cell(f['desc'])} |"
        )
    return "\n".join(lines)


def render_section_md(s: Section, level: int = 2) -> str:
    hashes = "#" * level
    title = str(s["title"])
    desc = str(s.get("desc") or "")
    fields = s.get("fields") or []
    example = textwrap.dedent(str(s.get("example") or "")).strip("\n")
    out = [f"{hashes} {title}", "", desc, "", md_table(fields)]
    if example:
        out += ["", "示例:", "", "```yaml", example, "```"]
    return "\n".join(out).rstrip() + "\n"


def render_full_markdown() -> str:
    intro = f"""# config.yaml 配置项参考

这份文档由 `scripts/config_help.py` 生成，用于查询 `runs/expNNN_<name>/config.yaml` 支持的字段。

常用命令:

```bash
python scripts/config_help.py                  # 列出配置点
python scripts/config_help.py eval.testsets    # 查看单个配置点
python scripts/config_help.py perf.scenes
python scripts/config_help.py --write CONFIG_REFERENCE.md
```

路径约定: `scripts/run_from_config.py` 读取路径字段时，相对路径默认相对项目根目录；支持 `${{ROOT}}` / `{{ROOT}}` / `${{EXP_DIR}}` / `{{EXP_DIR}}`。
""".rstrip()
    parts = [intro, ""]
    for s in SECTIONS:
        parts.append(render_section_md(s, level=2))
    return "\n".join(parts).rstrip() + "\n"


def render_section_text(s: Section) -> str:
    lines = [str(s["title"]), "=" * len(str(s["title"])), "", str(s.get("desc") or ""), ""]
    for f in s.get("fields") or []:
        lines.append(f"- {f['name']}")
        lines.append(f"    类型/可选值 : {f['type']}")
        lines.append(f"    默认/必填   : {f['default']}")
        lines.append(f"    说明       : {f['desc']}")
    example = textwrap.dedent(str(s.get("example") or "")).strip("\n")
    if example:
        lines += ["", "示例:", "", example]
    return "\n".join(lines).rstrip() + "\n"


def list_sections() -> str:
    lines = [
        "config.yaml 可配置点:",
        "",
    ]
    for s in SECTIONS:
        lines.append(f"  - {s['name']:<14} {s['title']}")
    lines += [
        "",
        "查看某个配置点:",
        "  python scripts/config_help.py eval.testsets",
        "  python scripts/config_help.py perf.scenes",
        "",
        "生成完整 Markdown:",
        "  python scripts/config_help.py --write CONFIG_REFERENCE.md",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("section", nargs="?", help="配置点名，如 model / eval / eval.testsets / perf.scenes")
    ap.add_argument("--markdown", action="store_true", help="输出完整 Markdown 到 stdout")
    ap.add_argument("--write", metavar="PATH", help="把完整 Markdown 写到指定文件")
    args = ap.parse_args()

    if args.write:
        out = Path(args.write)
        if not out.is_absolute():
            out = ROOT / out
        out.write_text(render_full_markdown(), encoding="utf-8")
        print(f"已写入 {out}")
        return

    if args.markdown:
        sys.stdout.write(render_full_markdown())
        return

    if args.section:
        sys.stdout.write(render_section_text(find_section(args.section)))
        return

    sys.stdout.write(list_sections())


if __name__ == "__main__":
    main()
