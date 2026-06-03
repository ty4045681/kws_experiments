# config.yaml 配置项参考

这份文档由 `scripts/config_help.py` 生成，用于查询 `runs/expNNN_<name>/config.yaml` 支持的字段。

常用命令:

```bash
python scripts/config_help.py                  # 列出配置点
python scripts/config_help.py eval.testsets    # 查看单个配置点
python scripts/config_help.py perf.scenes
python scripts/config_help.py --write CONFIG_REFERENCE.md
```

路径约定: `scripts/run_from_config.py` 读取路径字段时，相对路径默认相对项目根目录；支持 `${ROOT}` / `{ROOT}` / `${EXP_DIR}` / `{EXP_DIR}`。

## meta — 实验元信息

由 new_experiment.py 自动填一部分；用于 registry.csv / REPORT.md 展示。

| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |
|---|---|---|---|
| `exp_id` | str | 自动填，如 exp003 | 实验编号；必须形如 expNNN。 |
| `name` | str | 自动填 | 实验短名；会显示在报告中。 |
| `date` | str | 自动填 | 实验创建日期。 |
| `variable` | str | 可选 | 本次消融变量名，如 lr / threshold / chunk_size。 |
| `value` | str | 可选 | 变量取值；建议保持字符串，方便比较。 |
| `notes` | str | 可选 | 自由备注，会显示在实验元数据表中。 |
| `ckpt_path` | path | 可选 | checkpoint 路径，仅作记录。 |

示例:

```yaml
meta:
  exp_id: exp003
  name: lights_on_t035
  variable: threshold
  value: "0.35"
  notes: "fixed-text 正样本 + car_noise 负样本"
```

## model — sherpa-onnx 模型文件

运行 eval / sweep / perf 时必填。相对路径按项目根目录解析；也支持 ${ROOT} / ${EXP_DIR}。

| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |
|---|---|---|---|
| `tokens` | path | 必填 | tokens.txt。 |
| `encoder` | path | 必填 | encoder ONNX 文件。 |
| `decoder` | path | 必填 | decoder ONNX 文件。 |
| `joiner` | path | 必填 | joiner ONNX 文件。 |
| `keywords_file` | path | 必填 | sherpa-onnx keywords.txt；每行 '<tokens> @<phrase>'。 |

示例:

```yaml
model:
  tokens: sherpa_eval/model/tokens.txt
  encoder: sherpa_eval/model/encoder.onnx
  decoder: sherpa_eval/model/decoder.onnx
  joiner: sherpa_eval/model/joiner.onnx
  keywords_file: sherpa_eval/model/keywords.txt
```

## train — 训练超参记录

仅用于记录和横向对照，不会被 icefall 训练脚本直接读取。部分字段会平铺到 registry.csv。

| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |
|---|---|---|---|
| `stage` | str | pretrain | 训练阶段，如 pretrain / finetune。 |
| `base_model` | str | 可选 | 基础模型名。 |
| `subset` | str | 可选 | 训练数据子集。 |
| `bpe_size` | int | 可选 | BPE 大小。 |
| `num_epochs` | int | 可选 | 训练 epoch 数。 |
| `base_lr` | float | 可选 | 学习率。 |
| `lr_epochs` | float | 可选 | icefall run.sh 对应字段。 |
| `lr_batches` | int | 可选 | icefall run.sh 对应字段。 |
| `max_duration` | float/int | 可选 | 训练 max_duration。 |
| `use_fp16` | bool | 可选 | 是否使用 fp16。 |
| `causal` | bool | 可选 | 是否 causal。 |
| `num_encoder_layers` | str | 可选 | 结构记录，如 '1,1,1,1,1,1'。 |
| `feedforward_dim` | str | 可选 | 结构记录。 |
| `encoder_dim` | str | 可选 | 结构记录。 |
| `encoder_unmasked_dim` | str | 可选 | 结构记录。 |
| `decoder_dim` | int | 可选 | 结构记录。 |
| `joiner_dim` | int | 可选 | 结构记录。 |

示例:

```yaml
train:
  stage: finetune
  base_model: zipformer-3.3M
  base_lr: 0.0005
  num_epochs: 12
  causal: true
```

## decode — 解码/评测参数记录

用于记录 icefall decode / 评测口径；eval 中未填写时，run_from_config.py 会回退读取部分字段。

| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |
|---|---|---|---|
| `epoch` | int | 可选 | decode 使用的 epoch。 |
| `avg` | int | 可选 | checkpoint averaging 数。 |
| `chunk_size` | int | 可选 | decode chunk size。 |
| `left_context_frames` | int | 可选 | decode left context。 |
| `keywords_score` | float | 可选 | eval.keywords_score 为空时的回退值。 |
| `keywords_threshold` | float | 可选 | eval.keywords_threshold 为空时的回退值。 |
| `max_duration` | float/int | 可选 | decode max_duration。 |

示例:

```yaml
decode:
  epoch: 12
  avg: 2
  chunk_size: 16
  left_context_frames: 64
  keywords_score: 1.0
  keywords_threshold: 0.35
```

## onnx — 部署/导出记录

仅作记录，便于在 REPORT.md / registry.csv 中和性能结果对齐。

| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |
|---|---|---|---|
| `exported` | bool | false | 是否已导出 ONNX。 |
| `quant` | str | fp32 | 量化类型，如 fp32 / fp16 / int8。 |
| `chunk_size` | int | 可选 | 导出时 chunk size。 |
| `left_context_frames` | int | 可选 | 导出时 left context。 |

示例:

```yaml
onnx:
  exported: true
  quant: fp32
  chunk_size: 16
  left_context_frames: 128
```

## eval — 准确率评测总配置

run_from_config.py 的 manifest / eval / sweep stage 会读取这一段。

| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |
|---|---|---|---|
| `manifest_dir` | path | sherpa_eval/data | build_manifest.py 输出目录；REPORT.md 也默认从这里读 duration。 |
| `suffix` | str | onnx | metric 文件 backend 后缀，如 onnx / onnx-int8。 |
| `provider` | str | cpu | 传给 sherpa-onnx，如 cpu。 |
| `num_threads` | int | 2 | KeywordSpotter num_threads。 |
| `chunk_seconds` | float | 0.5 | 流式喂入 chunk 秒数。 |
| `keywords_threshold` | float | decode.keywords_threshold | KWS 阈值；越高越保守。 |
| `keywords_score` | float | decode.keywords_score | 关键词加成分。 |
| `num_trailing_blanks` | int | 可选 | 传给 sherpa-onnx KeywordSpotter。 |
| `max_active_paths` | int | 可选 | 传给 sherpa-onnx KeywordSpotter。 |
| `limit` | int/null | null | 调试时只跑前 N 条。 |
| `debug_lists` | bool | true | false 时 metric 文件不写 TP/FP/FN 明细列表。 |
| `test_only_keywords` | bool | false | true 时严格 ref == keyword；false 时 keyword in ref。 |
| `phrase_space_to_underscore` | bool | true | keywords phrase 和 trigger 空格转下划线。 |
| `thresholds` | list[str/float] | [] | sweep stage 使用；生成 suffix=onnx-tXX。 |
| `testsets` | list[dict] | [] | 测试集列表；详见 eval.testsets。 |
| `negative_hours` | dict | small/large 默认 | testset -> 负样本小时数，用于 FP 换算 FA/hour。 |
| `extra_args` | str/list | 可选 | 追加给 sherpa_onnx_kws_eval.py 的额外 CLI 参数。字符串按 shell 规则拆分。 |

示例:

```yaml
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
```

## eval.testsets[] — 准确率测试集配置

每个元素定义一个 testset。manifest stage 生成 JSONL；eval/sweep stage 消费 JSONL。

| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |
|---|---|---|---|
| `name` | str | 必填 | testset 名；会出现在 metric-<name>-onnx.txt 和报告中。 |
| `manifest` | path | 可选 | 已有 JSONL/TSV manifest；填了它可省略 mode/audio_dir。 |
| `mode` | transcript \| auto-pair \| fixed-text | manifest 为空时必填 | 决定 build_manifest.py 的转写来源。 |
| `audio_dir` | path | mode 需要 | 音频根目录。 |
| `transcript` | path | mode=transcript | 每行 '<utt_id> text' 的转写文件。 |
| `text / fixed_text` | str | mode=fixed-text | 所有音频共用文本；适合批量正样本。 |
| `ext` | str | .wav | 音频扩展名。 |
| `recursive` | bool | true | 是否递归扫描 audio_dir。false 会加 --no-recursive。 |
| `upper` | bool | true | 是否把 text 转大写。false 会加 --no-upper。 |
| `space_to_underscore` | bool | true | 是否把空格转下划线。false 会加 --no-space-to-underscore。 |
| `include_empty_text` | bool | false | 是否保留无转写音频。 |
| `id_mode` | stem \| relpath | relpath | utt_id 生成方式。递归目录建议 relpath。 |
| `extra_eval_args` | str/list | 可选 | 只追加给这个 testset 的 eval CLI 参数。 |

示例:

```yaml
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
```

## perf — 性能测试总配置

run_from_config.py 的 perf stage 会读取这一段。scene 内字段会覆盖 perf 总配置。

| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |
|---|---|---|---|
| `hardware` | str | 可选 | 硬件描述，仅作记录。 |
| `notes` | str | 可选 | 性能测试备注，仅作记录。 |
| `suffix` | str | eval.suffix/onnx | perf 文件 backend 后缀。 |
| `provider` | str | eval.provider/cpu | 传给 sherpa-onnx。 |
| `num_threads` | int | eval.num_threads/1 | 默认 spotter num_threads；scene 可覆盖。 |
| `chunk_seconds` | float | eval.chunk_seconds/0.5 | 默认 chunk 秒数；scene 可覆盖。 |
| `warmup` | int | 2 | 预热次数。 |
| `pacing` | full \| realtime | full | concurrent/batch_streaming/cpu_sweep 默认 pacing。 |
| `duration_seconds` | float | 30 | 持续压测时长；scene 可覆盖。 |
| `testset` | str | 可选 | 性能测试默认音频池，引用 eval.testsets[].name。 |
| `manifest` | path | 可选 | 性能测试默认 manifest；优先级高于 testset。 |
| `limit` | int | 可选 | 默认只取 manifest 前 N 条。 |
| `keywords_threshold` | float | eval/decode 回退 | 性能测试构建 spotter 的阈值。 |
| `keywords_score` | float | eval/decode 回退 | 性能测试构建 spotter 的关键词加成。 |
| `num_trailing_blanks` | int | 可选 | 传给 sherpa-onnx。 |
| `max_active_paths` | int | 可选 | 传给 sherpa-onnx。 |
| `scenes` | list[dict] | [] | 性能场景列表；详见 perf.scenes。 |
| `extra_args` | str/list | 可选 | 追加给所有 perf CLI 的额外参数。 |

示例:

```yaml
perf:
  testset: wakeword_pos
  suffix: onnx
  provider: cpu
  scenes:
    - scene: single_cpu1t
      mode: single
      num_threads: 1
```

## perf.scenes[] — 性能测试场景配置

每个元素会生成一份 perf-<scene>-<suffix>[-tag].json。字段按 mode 生效。

| 字段 | 类型/可选值 | 默认/是否必填 | 说明 |
|---|---|---|---|
| `scene` | str | 必填 | 场景名；用于文件名和报告列名。 |
| `mode` | single \| concurrent \| batch \| batch_streaming \| cpu_sweep | 必填 | 性能测试模式。 |
| `testset` | str | 可选 | 覆盖 perf.testset，引用 eval.testsets[].name。 |
| `manifest` | path | 可选 | 覆盖 perf.manifest。 |
| `suffix` | str | 继承 perf.suffix | backend 后缀。 |
| `tag` | str | 可选 | 追加到文件名，如 int8 / c8。 |
| `provider` | str | 继承 perf.provider | 传给 sherpa-onnx。 |
| `num_threads` | int | 继承 perf.num_threads | spotter num_threads。 |
| `chunk_seconds` | float | 继承 perf.chunk_seconds | chunk 秒数。 |
| `limit` | int | 继承 perf.limit | 只取前 N 条音频。 |
| `warmup` | int | 继承 perf.warmup | 预热次数。 |
| `keywords_threshold` | float | 继承 perf/eval/decode | 构建 spotter 的阈值。 |
| `keywords_score` | float | 继承 perf/eval/decode | 构建 spotter 的关键词加成。 |
| `extra_args` | str/list | 可选 | 只追加给该 scene 的额外 CLI 参数。 |
| `concurrency` | int | mode=concurrent/batch_streaming | 并发路数；batch_streaming 中也等价于活跃 stream 数。 |
| `duration_seconds` | float | mode=concurrent/batch_streaming/cpu_sweep | 持续压测秒数。 |
| `pacing` | full \| realtime | mode=concurrent/batch_streaming/cpu_sweep | full 跑满 CPU；realtime 模拟真实麦克风。 |
| `batch_size` | int | mode=batch | offline batch 大小。batch_streaming 可用 concurrency。 |
| `n_batches` | int | mode=batch | offline batch 轮数。 |
| `inner_mode` | concurrent \| batch_streaming | mode=cpu_sweep | cpu_sweep 内层模式。 |
| `concurrency_list` | str | mode=cpu_sweep | 逗号分隔并发点，如 '1,2,4,8,16'。 |
| `target_cpu` | float | mode=cpu_sweep | CPU 预算阈值。 |
| `cpu_budget_mode` | per_core \| total | mode=cpu_sweep | CPU% 口径。 |
| `cpu_affinity` | str | mode=cpu_sweep/Linux | 绑核，如 '0' / '0-3' / '0,2,4'。 |

示例:

```yaml
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
```
