# KWS 消融实验脚手架(icefall gigaspeech/KWS)

一套轻量、可扩展的实验管理工具,围绕 icefall `egs/gigaspeech/KWS` 的
`decode.py` 输出、sherpa-onnx 评测结果与推理性能测试设计。

- 支持**任意多个测试集**(small / large / 自定义场景集如 wakeword_pos /
  car_noise / kitchen / ...)
- 支持**任意多个性能测试场景**(单流时延 / 多路并发 / 批量解码)
- 准确率 与 性能 在同一份 `REPORT.md` 与同一张 `registry.csv` 里并排展示

## 目录结构

```
kws_experiments/
├── README.md                  # 本文件
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
│   ├── new_experiment.py      # 新建一次实验目录
│   ├── parse_decode.py        # metric-*.txt → metrics.json(顺便并 perf-*.json)
│   ├── parse_perf.py          # 单独收集 perf-*.json → metrics.json
│   ├── update_registry.py     # metrics.json → 三张 CSV
│   ├── build_report.py        # CSV → REPORT.md(含性能章节)
│   └── report.ipynb           # 分析 / 可视化模板
├── sherpa_eval/               # sherpa-onnx 评测端(独立可跑)
│   ├── run.sh                 # 多 testset 循环:build_manifest + eval
│   ├── sherpa_onnx_kws_eval.py
│   ├── build_manifest.py
│   └── README.md
└── sherpa_perf/               # sherpa-onnx 性能测试端(独立可跑)
    ├── run.sh                 # 多场景循环:single / concurrent / batch
    ├── sherpa_onnx_kws_perf.py
    └── README.md
```

## 完整工作流(每次新实验)

```bash
# 1. 新建一次实验目录(自动分配 expNNN)
python scripts/new_experiment.py --name lr5e-5 --variable lr --value 5e-5

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
python scripts/parse_decode.py runs/exp002_lr5e-5      # 同时收 perf-*.json
python scripts/update_registry.py runs/exp002_lr5e-5
python scripts/build_report.py
```

或者一键串起来:

```bash
scripts/run.sh --stage 0 --stop-stage 4
# 0:new  1:parse(含 perf)  2:register  3:report  4:rebuild-all
```

完成后:
- `registry.csv`、`per_command.csv`、`per_perf.csv` 都已追加该实验
- `REPORT.md` 自动出"准确率·总览"、"性能·总览"等章节
- 打开 `scripts/report.ipynb` 看 Recall-FA 散点、Per-keyword 热力图,
  以及性能的并发扫描曲线、延迟分布图

## sherpa-onnx 端

- **评测**:`sherpa_eval/run.sh` 用 `TESTSETS` 数组循环跑多个测试集,
  每个产出 `metric-<testset>-onnx.txt`。详见 `sherpa_eval/README.md`。
- **性能**:`sherpa_perf/run.sh` 用 `SCENES` 数组循环跑多个性能场景:
  - `single` — 单流时延 / RTF
  - `concurrent` — N 路并发(每线程独立 spotter),测吞吐 / P95 延迟
  - `batch` — `decode_streams` 批量解码,测服务端 batch 吞吐

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
