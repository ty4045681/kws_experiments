# KWS 消融实验脚手架(icefall gigaspeech/KWS)

一套轻量、可扩展的实验管理工具,围绕 icefall `egs/gigaspeech/KWS` 的
`decode.py` 输出、sherpa-onnx 评测结果与推理性能测试设计。

- 支持**任意多个测试集**(small / large / 自定义场景集如 wakeword_pos /
  car_noise / kitchen / ...)
- 支持**任意多个性能测试场景**(单流时延 / 多路并发 / 批量解码)
- 支持**配置驱动执行**:每个实验的 `config.yaml` 可以直接驱动
  manifest / eval / sweep / perf / 入库 / 报告,不用反复改 `run.sh`
- 准确率 与 性能 在同一份 `REPORT.md` 与同一张 `registry.csv` 里并排展示

## 目录结构

```
kws_experiments/
├── README.md                  # 本文件
├── pyproject.toml             # Python/uv 依赖声明
├── uv.lock                    # uv 锁文件,保证依赖可复现
├── CONFIG_REFERENCE.md        # config.yaml 可配置项完整参考 ← 可由 config_help.py 生成
├── registry.csv               # 所有实验的一行式汇总(宽表) ← 自动生成
├── per_command.csv            # 每实验×每关键词的准确率长表 ← 自动生成
├── per_perf.csv               # 每实验×每场景的性能长表 ← 自动生成
├── REPORT.md                  # 自动生成的人类可读总览(准确率+性能)
├── templates/
│   └── config.yaml            # 单次实验的配置模板
├── runs/
│   └── exp001_baseline/       # 每次实验一个目录
│       ├── config.yaml        # 这次实验的超参/变量
│       ├── metrics/           # 放两类输出:
│       │   ├── metric-small-pt.txt           # 准确率(icefall decode.py)
│       │   ├── metric-car_noise-onnx.txt     # 准确率(sherpa-onnx)
│       │   ├── perf-single_cpu1t-onnx.json   # 性能(sherpa-onnx 测试)
│       │   └── perf-concurrent_c8-onnx.json  # 性能(多场景)
│       └── metrics.json       # 解析后的结构化指标(含 runs 与 perf_runs)
├── scripts/                   # 总入口
│   ├── run.sh                 # 5 阶段总流程(新建/解析/入库/报告/重建)
│   ├── run_from_config.py     # 从 runs/expXXX/config.yaml 直接驱动 eval/perf/入库/报告
│   ├── config_help.py         # 查询/生成 config.yaml 配置项参考
│   ├── new_experiment.py      # 新建一次实验目录
│   ├── parse_decode.py        # metric-*.txt → metrics.json(顺便并 perf-*.json)
│   ├── parse_perf.py          # 单独收集 perf-*.json → metrics.json
│   ├── update_registry.py     # metrics.json → 三张 CSV
│   ├── build_report.py        # CSV → REPORT.md(含性能章节)
│   ├── bench_zipformer_streaming_onnx.py   # ONNX Runtime 流式 encoder 前向基准
│   ├── bench_zipformer_streaming_mindir.py # MindSpore Lite 流式 encoder 前向基准
│   └── report.ipynb           # 分析 / 可视化模板
├── sherpa_eval/               # sherpa-onnx 评测端(独立可跑)
│   ├── run.sh                 # 多 testset 循环:build_manifest + eval
│   ├── sherpa_onnx_kws_eval.py
│   ├── build_manifest.py
│   └── README.md
└── sherpa_perf/               # sherpa-onnx 性能测试端(独立可跑)
    ├── run.sh                 # 多场景循环:single/concurrent/batch/batch_streaming/cpu_sweep
    ├── sherpa_onnx_kws_perf.py
    └── README.md
```

## 环境准备(uv)

本仓库使用 `pyproject.toml` + `uv.lock` 管理 Python 依赖。首次使用时在仓库
根目录执行:

```bash
uv sync
```

这会安装运行评测和性能脚本所需的核心依赖:

- `sherpa-onnx`: sherpa-onnx KWS 推理与评测
- `numpy`: 音频数组和统计计算
- `psutil`: `cpu_sweep` 性能场景的 CPU 采样
- `pyyaml`: 优先用于读取 `config.yaml`；如果没有也有内置简化解析器兜底

如果还要运行 `scripts/report.ipynb` 里的分析和可视化模板,安装 notebook
可选依赖:

```bash
uv sync --extra notebook
```

常用脚本建议通过 `uv run` 执行,例如:

```bash
uv run python scripts/config_help.py
uv run python scripts/run_from_config.py runs/expNNN_<name>
```

## 完整工作流(每次新实验)

### 方式 A:配置驱动(推荐)

现在推荐把每次测试会变化的内容写进该实验自己的
`runs/expNNN_<name>/config.yaml`,不要反复改 `sherpa_eval/run.sh` /
`sherpa_perf/run.sh`。

```bash
# 1. 新建一次实验目录
uv run python scripts/new_experiment.py --name lr5e-5 --variable lr --value 5e-5

# 2. 编辑 runs/expNNN_lr5e-5/config.yaml:
#    - model.tokens / encoder / decoder / joiner / keywords_file
#    - eval.testsets
#    - eval.negative_hours
#    - perf.scenes(可选)

# 3. 一条命令跑 manifest → eval → perf → parse → register → report
uv run python scripts/run_from_config.py runs/expNNN_lr5e-5

# 如果只想看会执行哪些命令,先 dry-run
uv run python scripts/run_from_config.py runs/expNNN_lr5e-5 --dry-run

# 只跑准确率链路
uv run python scripts/run_from_config.py runs/expNNN_lr5e-5 \
  --stage manifest,eval,parse,register,report

# 只跑性能链路
uv run python scripts/run_from_config.py runs/expNNN_lr5e-5 \
  --stage perf,parse,register,report

# 只跑某个测试集 / 性能场景
uv run python scripts/run_from_config.py runs/expNNN_lr5e-5 \
  --stage manifest,eval --only-testset car_noise
uv run python scripts/run_from_config.py runs/expNNN_lr5e-5 \
  --stage perf --only-scene concurrent_c8
```

配置时如果忘了某个配置点支持哪些字段,可以直接查:

```bash
uv run python scripts/config_help.py                  # 列出所有配置点
uv run python scripts/config_help.py model            # 模型字段
uv run python scripts/config_help.py eval.testsets    # 测试集字段/模式
uv run python scripts/config_help.py perf.scenes      # 性能场景字段/模式
```

完整字段参考见 `CONFIG_REFERENCE.md`;如果改了脚本里的字段说明,可重新生成:

```bash
uv run python scripts/config_help.py --write CONFIG_REFERENCE.md
```

`config.yaml` 中可执行部分示例:

```yaml
model:
  tokens: sherpa_eval/model/tokens.txt
  encoder: sherpa_eval/model/encoder.onnx
  decoder: sherpa_eval/model/decoder.onnx
  joiner: sherpa_eval/model/joiner.onnx
  keywords_file: sherpa_eval/model/keywords.txt

eval:
  manifest_dir: sherpa_eval/data
  suffix: onnx
  provider: cpu
  num_threads: 2
  chunk_seconds: 0.5
  keywords_threshold: 0.35
  keywords_score: 1.0
  thresholds: ["0.20", "0.25", "0.30", "0.35"]
  testsets:
    - name: wakeword_pos
      mode: fixed-text
      audio_dir: /data/wakeword_pos
      text: "lights on"
    - name: car_noise
      mode: transcript
      audio_dir: /data/car_noise
      transcript: /data/car_noise.text
  negative_hours:
    wakeword_pos: 0
    car_noise: 2.1

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
```

支持的 stage:

| stage | 作用 |
|---|---|
| `manifest` | 根据 `eval.testsets` 调 `sherpa_eval/build_manifest.py` |
| `eval` | 根据 `eval.testsets` 调 `sherpa_eval/sherpa_onnx_kws_eval.py` |
| `sweep` | 根据 `eval.thresholds` 做阈值扫描,产 `onnx-tXX` 指标 |
| `perf` | 根据 `perf.scenes` 调 `sherpa_perf/sherpa_onnx_kws_perf.py` |
| `parse` | 调 `scripts/parse_decode.py` 生成 `metrics.json` |
| `register` | 调 `scripts/update_registry.py` 更新三张 CSV |
| `report` | 调 `scripts/build_report.py` 生成 `REPORT.md` |

`--stage all` 是默认值,等价于 `manifest,eval,perf,parse,register,report`;
`--stage full` 会额外包含 `sweep`。

路径约定:相对路径默认相对**项目根目录**,也可以使用 `${ROOT}` /
`${EXP_DIR}` 占位。

### 方式 B:手动产物收纳

```bash
# 1. 新建一次实验目录(自动分配 expNNN)
uv run python scripts/new_experiment.py --name lr5e-5 --variable lr --value 5e-5

# 2. 跑你自己的训练 + decode(参考 icefall run.sh)
#    把 metric-*.txt / perf-*.json 拷到该实验目录下,例如:
#    runs/exp002_lr5e-5/metrics/metric-small-pt.txt
#    runs/exp002_lr5e-5/metrics/metric-small-onnx.txt          # 可选
#    runs/exp002_lr5e-5/metrics/metric-car_noise-onnx.txt      # 可选,任意名
#    runs/exp002_lr5e-5/metrics/perf-single_cpu1t-onnx.json    # 可选,sherpa_perf 产出
#    runs/exp002_lr5e-5/metrics/perf-concurrent_c8-onnx.json   # 可选,多场景
#
#    命名约定:
#       准确率 metric-<testset>-<backend>[-tag].txt    (backend ∈ {pt, onnx})
#       性能   perf-<scene>-<backend>[-tag].json
#    tag 可选,会合并进 backend 名(如 onnx-t0.35 / onnx-int8)

# 3. 编辑该实验的 config.yaml,填入超参 / 训练备注 / 各 testset 的总时长

# 4. 解析并入库(stage 2/3/4)
uv run python scripts/parse_decode.py runs/exp002_lr5e-5      # 同时收 perf-*.json
uv run python scripts/update_registry.py runs/exp002_lr5e-5
uv run python scripts/build_report.py
```

或者一键串起来:

```bash
bash scripts/run.sh --stage 1 --stop-stage 1 --name lr5e-5
bash scripts/run.sh --stage 2 --stop-stage 4 --exp-dir runs/expNNN_lr5e-5
# 1:new  2:parse(含 perf)  3:register  4:report  5:rebuild-all
```

完成后:
- `registry.csv`、`per_command.csv`、`per_perf.csv` 都已追加该实验
- `REPORT.md` 自动出"准确率·总览"、"性能·总览"等章节
- 打开 `scripts/report.ipynb` 看 Recall-FA 散点、Per-keyword 热力图,
  以及性能的并发扫描曲线、延迟分布图

## 流式 Zipformer encoder 前向基准

`scripts/bench_zipformer_streaming_onnx.py` 和
`scripts/bench_zipformer_streaming_mindir.py` 用于直接测量已导出的 streaming
Zipformer encoder。它们使用随机 fbank 特征，第一步将所有 cache 和
`processed_lens` 置零，后续步骤将 `new_*` 输出状态回填为下一次输入。

计时范围仅包含一次模型前向调用：ONNX Runtime 的 `session.run()` 或
MindSpore Lite 的 `model.predict()`。特征窗口切片、输入设置、cache 回填和
warmup 均不计入延迟。因此该结果用于比较 Python API 下的逐 chunk forward
延迟，而不是端到端音频处理时间或纯 C++ kernel 时间。

默认配置匹配 `chunk_size=16`、`left_context_frames=64` 的模型：单线程、绑定
Linux CPU 0、warmup 20 步、计时 100 步。输入窗口为 39 帧 80 维 fbank；每步
前移 32 帧，即 320 ms 音频。缓存将在第 4 步后填满。

```bash
# ONNX Runtime: requires onnxruntime in the active Python environment
uv run --with onnxruntime python scripts/bench_zipformer_streaming_onnx.py \
  --model /path/to/encoder.onnx

# MindSpore Lite: install the platform-matched mindspore_lite package first
uv run python scripts/bench_zipformer_streaming_mindir.py \
  --model /path/to/encoder.mindir
```

常用可配置参数为 `--chunk-size`、`--left-context-frames`、`--warmup`、
`--loops`、`--threads`、`--cpu` 和 `--no-cpu-bind`。脚本会校验 encoder 的输入
窗口是否符合 `T = 2 * chunk_size + 7`；若导出模型的 chunk size 不同，必须通过
`--chunk-size` 显式指定导出时的值。

## sherpa-onnx 端

日常建议优先用上面的 `scripts/run_from_config.py`。下面两个目录仍然可以
作为独立工具直接运行,也方便单独调试 manifest / eval / perf。

- **评测**:`sherpa_eval/run.sh` 用 `TESTSETS` 数组循环跑多个测试集,
  每个产出 `metric-<testset>-onnx.txt`。详见 `sherpa_eval/README.md`。
- **性能**:`sherpa_perf/run.sh` 用 `SCENES` 数组循环跑多个性能场景:
  - `single` — 单流时延 / RTF
  - `concurrent` — N 路并发(每线程独立 spotter),测吞吐 / P95 延迟
  - `batch` — `decode_streams` 批量解码,测服务端 batch 吞吐
  - `batch_streaming` — B 路真实流式 batch,测共享 spotter 的尾延迟 / 吞吐
  - `cpu_sweep` — 扫描并发点并采样 CPU%,估算预算内最大并发

  每个场景产出 `perf-<scene>-<backend>[-tag].json`。详见 `sherpa_perf/README.md`。

两个工具都默认把产物直接写到 `runs/expNNN_*/metrics/`,所以跑完就立即
被脚手架收纳。

## 设计要点

- **单一事实来源**:`metrics.json` 是结构化真值,三张 CSV 由它生成,
  从不手工编辑 CSV
- **宽表 + 长表 并存**:`registry.csv` 一行一个实验适合横向看;
  `per_command.csv` 一行一个 keyword×指标;`per_perf.csv` 一行一个场景×指标
- **PT 与 ONNX 并列**:每次实验同时记录两套准确率指标,便于定位
  "训练问题还是部署链路问题"
- **准确率与性能上同一张表**:同一个 `exp_id` 下的训练/评测/推理性能全部绑定
- **任意 testset 与 任意 scene**:解析、入库、报告、Notebook 都按文件名
  动态发现,新增只需多放一个 metric-*.txt 或 perf-*.json
- **追加,不覆盖**:重复运行 `update_registry.py` 按 exp_id 更新而不破坏其它行
- **幂等**:任何一步都可以重跑

## 关键约定

- `exp_id` 必须形如 `exp001`、`exp002` 三位数字
- 准确率文件:`metric-<testset>-<backend>[-tag].txt`
  - `backend` ∈ {`pt`, `onnx`};`tag` 可选(如阈值扫描 `-t0.35`)
- 性能文件:`perf-<scene>-<backend>[-tag].json`
  - `scene` 任意(`single_cpu1t` / `concurrent_c8` / `batch_b16` / ...)
- 每个实验的 `config.yaml` 用 `eval.negative_hours` 字典声明每个 testset
  的总时长(小时),用于把 FP 数换算成 FA/hour,例如:

  ```yaml
  eval:
    negative_hours:
      small: 40
      large: 23
      car_noise: 5.2
  ```

  默认值:small=40h、large=23h(见 icefall RESULTS.md)。新场景集请自己测量。
  为向后兼容,也仍然识别旧的扁平键 `negative_hours_<testset>`。
