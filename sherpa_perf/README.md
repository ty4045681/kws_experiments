# sherpa_perf —— sherpa-onnx KWS 推理性能测试

测三类东西:

| 类别  | 关注              | 用哪个 mode             |
|-------|-------------------|-------------------------|
| 时延  | 单条音频处理多快  | `single`                |
| 并发  | 能稳定接住几路    | `concurrent`            |
| 吞吐  | batch 服务上限    | `batch`                 |

输出每个场景一份 JSON,直接落到上层脚手架某次实验的 `metrics/` 目录,
被 `scripts/parse_perf.py` 收纳入库,在 `registry.csv` 和 `REPORT.md`
里和准确率指标并排展示。

## 目录

```
sherpa_perf/
├── README.md
├── run.sh                      # 一键多场景
└── sherpa_onnx_kws_perf.py     # 测试核心
```

## 快速上手

```bash
# 1. 把模型放到 sherpa_perf/model/(或在 run.sh 顶部改路径)
# 2. 准备 manifest(可直接复用 sherpa_eval 产物)
ls ../sherpa_eval/data/wakeword_pos.jsonl

# 3. 在 run.sh 顶部 SCENES 数组配置你要跑的场景,然后:
bash run.sh

# 4. 产出:
ls ../runs/exp001_baseline/metrics/perf-*.json
```

## 三种模式

### `single` — 单流时延 / RTF

单线程顺序处理 N 条音频,统计每条端到端 wall-clock 时间。
输出:延迟分布(P50/P90/P95/P99)、RTF(实时倍速)、平均吞吐。

```bash
python sherpa_onnx_kws_perf.py \
    --tokens model/tokens.txt --encoder ... --decoder ... --joiner ... \
    --keywords-file model/keywords.txt \
    --manifest ../sherpa_eval/data/wakeword_pos.jsonl \
    --mode single --limit 50 --num-threads 1 \
    --scene single_cpu1t --output-dir ../runs/exp001_baseline/metrics
```

适合回答:"一条 3 秒的录音从送入到拿到 trigger 要多久?"

### `concurrent` — 多路并发

启 N 个工作线程,每个线程**独立** `KeywordSpotter` 实例(贴近真实多路部署),
从共享音频池循环取数据,跑满 `--duration-seconds` 后停止。

```bash
python sherpa_onnx_kws_perf.py ... \
    --mode concurrent --concurrency 8 --duration-seconds 30 \
    --pacing full \
    --scene concurrent_c8 --output-dir ...
```

- `--pacing full`:不 sleep,跑满 CPU,看吞吐上限
- `--pacing realtime`:按音频时长 sleep,模拟"8 个用户同时讲话",
  这种情况下 P95 延迟代表真实部署的尾延迟

适合回答:"这台机器能稳定承载多少路并发,P95 还可控?"

### `batch` — 批量解码

使用 sherpa-onnx 的 `decode_streams([s1, ..., sB])` 批处理 API。
每个 batch 同时创建 B 条 stream,全量喂入,然后反复批解码到收敛。

```bash
python sherpa_onnx_kws_perf.py ... \
    --mode batch --batch-size 16 --n-batches 20 \
    --scene batch_b16 --output-dir ...
```

适合回答:"服务端 batch_size=N 时,等效每条延迟和总吞吐是多少?"

## 输出 JSON 字段说明

每份 `perf-<scene>-<backend>[-tag].json` 顶层:

```json
{
  "mode": "concurrent",
  "scene": "concurrent_c8",
  "backend": "onnx",
  "tag": "",
  "chunk_seconds": 0.5,
  "n_manifest_samples": 50,
  "env": {
    "platform": "Linux-...",
    "cpu_count": 16,
    "cpu_model": "Intel(R) Xeon(R) ...",
    "provider": "cpu",
    "num_threads": 1,
    "sherpa_onnx_version": "1.x.y"
  },
  "model": {
    "encoder": "model/encoder-....onnx",
    "keywords_threshold": 0.35
  },
  "result": {
    // mode-specific 字段,见下
  }
}
```

### `result` 在三种 mode 下的字段

#### single
| 字段                              | 含义                                |
|-----------------------------------|-------------------------------------|
| `throughput_audio_per_wall`       | 音频秒数 / wall 秒,xRT,越大越好    |
| `latency_seconds.{p50,p95,p99}`   | 端到端延迟分布                       |
| `rtf.mean`                        | 平均 RTF(<1 即比实时快)             |

#### concurrent
| 字段                                | 含义                                  |
|-------------------------------------|---------------------------------------|
| `concurrency`                       | 并发路数                              |
| `pacing`                            | full / realtime                       |
| `throughput_calls_per_sec`          | 每秒完成多少条调用                    |
| `throughput_audio_per_wall`         | 总音频秒数 / wall,xRT                |
| `latency_seconds.{p50,p95,p99}`     | 单路端到端延迟分布                    |
| `rtf_per_stream.{mean,p95}`         | 单流 RTF(>1 表示掉队)                |

#### batch
| 字段                                | 含义                                  |
|-------------------------------------|---------------------------------------|
| `batch_size` / `n_batches`          | 批参数                                |
| `throughput_calls_per_sec`          | 调用吞吐                              |
| `batch_wall_seconds.{mean,p95}`     | 单 batch 总耗时                       |
| `per_call_latency_seconds.{mean,p95}` | 等效每条延迟 = batch 耗时 / batch_size |
| `batch_rtf.mean`                    | batch 音频总时长 vs wall 的比         |

## 文件命名约定

```
perf-<scene>-<backend>[-tag].json
```

- `scene`:任意名(`single_cpu1t` / `concurrent_c8` / `batch_b16` / ...)
- `backend`:一般是 `onnx`
- `tag`:可选,常用于区分配置变体(`-realtime` / `-int8` / ...)

上层 `scripts/parse_perf.py` 会按此模式发现并解析。

## 常见问题

- **为什么 concurrent 用每线程独立 spotter,而 batch 用共享 spotter?**
  - 真实部署里每路通常是独立的解码状态机(stream),用独立 spotter 更接近;
  - batch 模式正是要利用 sherpa-onnx 的批解码 API,必须共享同一个 spotter。
- **`--pacing full` vs `realtime` 该选哪个?**
  - 容量评估(找拐点) → `full`
  - 评估在线服务真实尾延迟 → `realtime`
- **要不要先 warmup?**
  - 默认 `--warmup 2`,跑两条丢弃,避免首条 cold-start 拖偏 P50。
