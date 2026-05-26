# sherpa_perf — sherpa-onnx KWS 推理性能测试

测的是**速度与容量**（延迟分布、RTF、并发承载、批吞吐），不测准确率（那是 `sherpa_eval/` 的工作）。
每个场景产出一份 `perf-<scene>-<backend>[-tag].json`，落入上层实验的 `metrics/`，
由 `scripts/parse_perf.py` 收纳进 `registry.csv` 与 `REPORT.md`，与准确率指标并排展示。

```
sherpa_perf/
├── README.md
├── run.sh                      # 多场景一键脚本
└── sherpa_onnx_kws_perf.py     # 测试核心
```

---

## 1. 模式概览

| Mode              | 回答的问题                                                  | 关键产出                                            |
|-------------------|-------------------------------------------------------------|-----------------------------------------------------|
| `single`          | 单条音频端到端处理多久？是否能跑实时？                      | `latency_seconds.{p50,p95,p99}`、`rtf.mean`         |
| `concurrent`      | 并发 N 路时尾延迟还稳吗？这台机器顶几路？                   | `latency_seconds.p95`、`rtf_per_stream.p95`         |
| `batch`           | 服务端凑齐 B 条音频后批量推理，吞吐上限是多少？             | `throughput_calls_per_sec`、`batch_rtf`             |
| `batch_streaming` | B 路真实实时流共享同一次 batch forward，单条 SLO 怎样？     | `latency_seconds.p95`、`throughput_calls_per_sec`   |
| `cpu_sweep`       | 在 X% CPU(单核/多核)预算下，最多撑住多少路并发？画出曲线。  | `sweep_points[]`、`max_concurrency_under_budget`、配对 CSV |

设计原则：**真实部署里每路通常是独立解码状态机**，所以 `concurrent` 给每个 worker 独立 `KeywordSpotter`；
`batch` 与 `batch_streaming` 都要利用 sherpa-onnx 的 `decode_streams([...])` 批解码 API，必须共享同一个 spotter，差别只在喂入节奏：
`batch` 全量喂完一批再解，`batch_streaming` 是 B 路时间片交错喂入、每 tick 一次 batch forward。

---

## 2. 关键指标定义

### 2.1 端到端延迟 `latency_seconds`

**计时边界**：`decode_one_blocking` 内 `t0 = perf_counter()`（line 173）到 `return perf_counter() - t0`（line 191）之间的 wall-clock。

**计入**：
- 主循环：所有 chunk 的 `accept_waveform` → `decode_stream` → `get_result` → 命中时的 `reset_stream`
- 尾部静音（`_tail_padding`，0.66s）的喂入与解码
- `input_finished()` 后的 drain 循环

**不计入**：
- `build_spotter` 模型加载（启动阶段一次性开销）
- WAV 解码与文件 I/O（`_preload_pool` 在计时前完成）
- warmup 调用（默认 2 次，结果丢弃；line 225-227 / 300-302）

**KWS 特异点**：脚本测的是"整条音频跑完"的延迟，**不是"keyword 命中即结束"的延迟**。命中后 `reset_stream` 继续监听，所以即便 keyword 出现在 1.2s 处，wall 仍包含剩余 ~1.8s + tail 的处理。这是有意设计——测稳态推理开销，避免命中位置造成的方差。如需测 "wake-to-response" 延迟，应另外实现：`get_result` 命中即 break，单独统计。

**百分位选择**：mean 会被长尾掩盖，统一汇报分位数：

| 字段  | 工程含义                                                    |
|-------|-------------------------------------------------------------|
| `p50` | 典型体验。一半请求快于此值。                                |
| `p90` | 容量规划用。                                                |
| `p95` | **SLO 主指标**。常作为是否可上线的硬门槛。                  |
| `p99` | 高可用服务必看；体现 GC / 抢占 / 线程切换造成的尾噪。       |
| `max` | 最坏个例，用于诊断 cold-start 或异常长样本。                |

注：`run_concurrent` 的 `rtf_per_stream` 不汇报 `p99` / `max`（line 341 的 `with_p99=False, with_max=False`），并发尾噪由 `latency_seconds.p99` 承担即可。

### 2.2 实时因子 RTF

```
RTF = wall_seconds / audio_seconds
```

| RTF        | 含义                                                          |
|------------|---------------------------------------------------------------|
| `< 1`      | 比实时快，能跟住麦克风流。健康 KWS 通常应 `< 0.3`。           |
| `≈ 1`      | 刚好跟上，无 headroom，并发场景几乎必然掉队。                 |
| `> 1`      | 处理慢于输入；流式缓冲单调增长，最终会 OOM 或丢音。           |

**`pacing` 与 RTF**：`realtime` 下 wall 包含 sleep 时间，所以 `rtf_per_stream ≈ 1 + 算力开销 / 音频时长`。这就是为何 realtime 下 `rtf_per_stream.p95 > 1.3` 是危险信号——sleep 都压不住算力开销，缓冲在涨。

### 2.3 xRT — `throughput_audio_per_wall`

```
xRT = audio_seconds_total / wall_seconds
```

直译："1 秒墙钟能消化多少秒音频"，也称**等效实时倍速**。与单流 RTF 的关系：

| 模式         | 关系                                                          |
|--------------|---------------------------------------------------------------|
| `single`     | `xRT ≈ 1 / rtf.mean`（顺序无并行，纯算力倒数视角）            |
| `concurrent` | `xRT ≈ N × (1 / rtf_per_stream)`，**理论上限 = 并发路数 N**   |
| `batch`      | `xRT ≈ 1 / batch_rtf`，B 条流的并行加速已折进 batch_rtf       |

**具体数字**（沿用第 3 节的示例）：

| 场景                       | rtf 视角            | xRT      | 解读                                           |
|----------------------------|---------------------|----------|------------------------------------------------|
| `single_cpu1t`             | `rtf.mean=0.027`    | 35.7     | 单流时 CPU 极空闲，硬件还有大量富余            |
| `concurrent_c8` `realtime` | `rtf_per_stream≈1.0`| 7.9      | 每路只跑实时，总 xRT ≈ N，符合"8 路同时讲话"   |
| `concurrent_c8` `full`     | `rtf_per_stream<1`  | 25~30    | 跑满 CPU 的真实吞吐拐点                        |
| `batch_b8`                 | `batch_rtf=0.0026`  | 384      | batch 摊薄推理常数开销，xRT 远高于 single      |

**工程意义**：xRT 是"一台机器能顶几路实时麦克风"的硬上限。部署留 20–30% buffer：`xRT_full = 25` → 上线最多开 ~18 路。

### 2.4 cps — `throughput_calls_per_sec`

```
cps = n_calls_total / wall_seconds
```

直译："1 秒墙钟能完成多少次完整调用"。与 xRT 的换算：

```
cps ≈ xRT / 平均音频时长
```

例：`xRT = 384`、平均 3s 音频 → `cps ≈ 128`，与 `batch_b8` 实测一致。

**工程意义**：服务端 QPS 容量规划的核心指标。前端 QPS 需求 1000、单机 cps = 128 → 至少 8 台（未含 buffer 与故障冗余）。

**为什么 `single` 不报 cps**：单线程顺序处理时 cps 与样本时长强耦合（同样的算力跑全 1s 样本 vs 全 5s 样本，cps 差 5 倍），意义有限；xRT 与样本时长无关，更能反映算力本身。看 `_summarize_single`（line 269）——只输出 `throughput_audio_per_wall`，没有 cps。

### 2.5 三者关系速查

```
RTF (单条)   ──取倒数──→  xRT (单流)
                              │
                         × 并行路数(concurrent) / batch 内并行(batch)
                              ↓
                          xRT (聚合)   ──÷ 平均音频时长──→  cps
```

回答不同问题用不同视角：

| 问题                                | 看哪个                               |
|-------------------------------------|--------------------------------------|
| 单条响应快不快？                    | `latency_seconds.p95`                |
| 单条算得过来吗？（是否能跑实时）    | `rtf.mean < 1`，理想 `< 0.3`         |
| 一台机器顶几路实时？                | `xRT_full` × 0.7~0.8（留 buffer）    |
| 服务端单机 QPS 上限？               | `cps`（来自 `concurrent_full` 或 `batch`）|

### 2.6 Pacing（`concurrent` 独有）

| `--pacing`   | 行为                                                  | 用途                                         |
|--------------|-------------------------------------------------------|----------------------------------------------|
| `full`       | 不 sleep，连续喂                                      | 找吞吐拐点（极限容量评估）。                 |
| `realtime`   | 每喂一 chunk `sleep(len(seg)/sr)`（line 182-183）     | 模拟 N 路真实麦克风，**P95 才有 SLO 意义**。 |

容量评估用 `full`；上线验证尾延迟用 `realtime`。

---

## 3. 音频处理流程

后续所有时间轴均以 **3s / 16 kHz / 单声道 wav，`chunk_seconds=0.5`** 为例：

- `chunk = 8000 samples`（一片 0.5s）→ 主体切 **6 个 chunk**
- 尾部 `_TAIL_SECONDS = 0.66s`（line 149）→ 再补 ~2 个 chunk，让最后一个词解码出来
- 数字仅为示意，实际值与机器/模型相关

### 3.1 共同前置

```
load_manifest()       JSONL → List[Sample]，顺手 wave.open 取时长
        ↓
build_spotter()       加载 tokens/encoder/decoder/joiner/keywords
        ↓
_preload_pool()       所有 wav 一次性 decode 为 float32 numpy；
                      避免 I/O 进入热路径计时
        ↓
warmup                每个 spotter 跑 N 条（默认 2），结果丢弃
```

### 3.2 `single` 模式时间轴

`run_single` (`sherpa_onnx_kws_perf.py:216-239`) 在主线程上对 pool 顺序遍历，
每条音频走 `decode_one_blocking(pacing="full")`：

```
T = 0.000s  s = kws.create_stream()
T = 0.000s  t0 = perf_counter()                ← 计时开始
            ┌────────── 主循环：6 个 chunk ─────────┐
T ≈ 0.000s  │ accept_waveform(audio[0:8000])       │
            │ while is_ready: decode_stream;       │
            │   get_result; 命中则 reset_stream    │
T ≈ 0.012s  │ accept_waveform(audio[8000:16000])   │
T ≈ 0.025s  │ ...                                  │
T ≈ 0.064s  │ accept_waveform(audio[40000:48000])  │
            └──────────────────────────────────────┘
T ≈ 0.064s  accept_waveform(tail)               ← 0.66s 静音
T ≈ 0.064s  input_finished()
T ≈ 0.064s  drain：while is_ready: decode_stream
T = 0.078s  return wall = perf_counter() - t0   ← 0.078s
```

记录到 `PerCallRecord(audio_seconds=3.0, wall_seconds=0.078, rtf=0.026)`。
跑完 N 条后 `_summarize_single` (line 269) 汇总。

**要点**：

- 流式语义：每片喂入立刻 decode，不等积攒。
- 命中后 `reset_stream` 不是终止，而是清状态继续监听（防止同句多次触发）。
- `pacing="full"`，wall = 纯计算耗时。

### 3.3 `concurrent` 模式时间轴

`run_concurrent` (line 283-342) 拓扑：

```
主线程
 ├─ _preload_pool()            音频池（共享只读）
 ├─ stop_flag = Event()
 ├─ per_thread_recs = [[]]*N   各线程独立列表，避免锁竞争
 ├─ 启动 N 个 worker(daemon=True)
 ├─ sleep(duration_seconds)
 ├─ stop_flag.set()
 └─ join 所有 worker
```

单个 worker 内（`worker`，line 298-313）；以 `--concurrency 8 --pacing realtime` 为例：

```
T = 0.0s   kws = factory()                ← 每线程独立 KeywordSpotter
T < 1s     warmup（pacing=full，结果丢弃）
T ≈ 1s     循环：while not stop_flag.is_set():
           ┌────────── 一次调用 (pacing=realtime) ──────────┐
           │ accept_waveform(chunk[0]); decode_stream       │
           │ sleep(0.5)              ← ★ 模拟麦克风节奏     │
           │ accept_waveform(chunk[1]); decode_stream       │
           │ sleep(0.5)                                     │
           │ ... 共 6 次 喂+sleep                           │
           │ accept_waveform(tail); input_finished; drain   │
           └────────────────────────────────────────────────┘
           wall ≈ 3.05s（≈ 音频时长 + 计算开销）
           记录 PerCallRecord; i += 1
T = 30s    stop_flag 被主线程置位 → 循环退出 → 线程结束
```

8 个 worker 并行执行同一段逻辑，被各自的 `sleep` 错开 → 任意时刻都有多个 stream
在 CPU 上推进，等价于 8 个用户同时讲话。聚合时把 8 个 thread 的 records 拼起来出统计。

**`pacing` 切换的语义差异**：

- `full`：wall 反映纯算力，`latency_seconds` 不再代表用户等待时间，而是 "在 N 路抢资源时这条流要多久"。容量评估用。
- `realtime`：wall ≈ 音频时长 + 算力开销。`rtf_per_stream.p95` 是关键 SLO；
  若 > 1.3，说明缓冲在涨，再加路就会崩。

### 3.4 `batch` 模式时间轴

`run_batch` (line 345-418) 单个 batch（`--batch-size 8`）三阶段：

```
T = 0.000s  阶段 A：建 stream
            streams = [kws.create_stream() for _ in range(8)]

T = 0.001s  阶段 B：全量喂入（尚未 decode）
            for i in 0..7:
                for chunk in audio_i: streams[i].accept_waveform(...)
                streams[i].accept_waveform(tail)
                streams[i].input_finished()

T = 0.005s  阶段 C：批解码循环（line 375-383）
            while True:
                ready = [s for s in streams if kws.is_ready(s)]
                if not ready: break
                kws.decode_streams(ready)        ← ★ 一次 forward 跑 ready 中所有 stream
                for s in ready:
                    r = kws.get_result(s)
                    if r: kws.reset_stream(s)
            # 各 stream 可能在不同轮次提前完成,ready 集合逐步收缩

T = 0.062s  return wall = 0.062, audio_total = 24.0  (8 × 3s)
```

与 `single` 的本质差异：

| 维度        | `single`              | `batch`                                           |
|-------------|-----------------------|---------------------------------------------------|
| 喂/算节奏   | 每片立刻 decode       | 全量喂完再批解码                                  |
| 模型调用    | `decode_stream(s)`    | `decode_streams([s1..sB])`                        |
| 加速来源    | —                     | 矩阵并行：一次 forward 的耗时 ≪ B × 单次 forward |

跑 `n_batches` 个 batch 后汇总：

- `per_call_latency_seconds = batch_wall / batch_size` —— **摊销值**，不是用户等待时间。
- 用户真实等待 = 单 batch wall + 凑齐 batch 的排队时间。
- 因此：**端侧实时唤醒禁用 batch；服务端语音网关用 `batch` 看吞吐上限，用 `batch_streaming` 看真实 SLO。**

### 3.5 `batch_streaming` 模式时间轴

`run_batch_streaming` 是"batch 但流式"——和 `batch` 一样共享单个 `KeywordSpotter`、用 `decode_streams([...])` 一次跑多条流；但和 `batch` 不同，B 条流在时间轴上是**交错的**，不是"全喂完再批解"。这是服务端语音网关真实部署的形态。

主循环每个 tick 做四件事（假设 `--concurrency 8 --pacing realtime`、3s 音频）：

```
T = 0.000s  tick_start
            ┌────────── (a) 喂 ──────────────────────────────────┐
            │ for st in active (B=8):                            │
            │   若还有 chunk:    accept_waveform(0.5s)           │
            │   否则若没喂 tail: accept_waveform(0.66s 静音)     │
            │   否则若没 finish: input_finished()                │
            └────────────────────────────────────────────────────┘
            ┌────────── (b) 批解码 ──────────────────────────────┐
            │ while True:                                        │
            │   ready = [st.s for st in active if is_ready(s)]   │
            │   if not ready: break                              │
            │   kws.decode_streams(ready)   ← ★ 一次 forward     │
            │   for st: get_result; 命中则 reset_stream          │
            └────────────────────────────────────────────────────┘
            ┌────────── (c) 回收 / 替换 ─────────────────────────┐
            │ for st in active:                                  │
            │   if finished_input and not is_ready(s):           │
            │     记录 wall = now - st.t_start                   │
            │     active[i] = _new_slot()  ← ★ 立刻补一条新流    │
            └────────────────────────────────────────────────────┘
T = 0.5s    (d) realtime 时 sleep 把 tick 凑到 chunk_seconds (0.5s)
            进入下一个 tick
```

**完成判定**：与 `run_batch` 主循环 break 条件一致 —— `input_finished and not is_ready(s)` 表示输入已关闭且 buffer 已 drain 完，此时这条流"用完了"，立即被新流替换以维持稳态 B 条活跃。

**单条 wall 定义**：`now - st.t_start`，其中 `t_start` 是 `_new_slot()` 创建该 slot 的瞬间。整段语义与 `decode_one_blocking` 的 `t0` 对齐——首片喂入到 drain 完成的真实墙钟。这也就是 **`latency_seconds.p95` 可以当 SLO 看的原因，不是 `batch_wall / batch_size` 那种摊销值**。

**与 `batch` 的差别一图概括**：

| 维度                  | `batch` (offline)             | `batch_streaming` (online)                                              |
|-----------------------|-------------------------------|-------------------------------------------------------------------------|
| 喂入节奏              | 一条流一次性喂完整段          | B 条流每 tick 各喂一片                                                  |
| `decode_streams` 调用 | 喂完之后反复 drain 直到空     | 每 tick 一次（内含 drain 循环）；流之间始终保持"差不多一样多缓冲"        |
| 单条延迟来源          | `batch_wall / B`（摊销）      | 真实 wall = chunk 数 × tick 时间 + 尾部 drain                            |
| 适用结论              | 服务端凑齐 B 条音频后批解码   | 服务端 B 路实时麦克风共享 batch forward                                  |

**pacing 语义**：和 `concurrent` 完全一致 ——

- `realtime`：每 tick 等到 `chunk_seconds`，wall ≈ 音频时长 + 算力开销 + tick 调度抖动。`latency_seconds.p95` 是 SLO；若 `rtf_per_stream.p95 > 1.3`，说明缓冲在涨，再加路就会崩。
- `full`：不 sleep，跑满 CPU 找吞吐拐点。`throughput_audio_per_wall` / `throughput_calls_per_sec` 是关键。

### 3.6 四种模式音频路径对比

| 阶段          | `single`                          | `concurrent` (`realtime`)                 | `batch`                                  | `batch_streaming` (`realtime`)             |
|---------------|-----------------------------------|-------------------------------------------|------------------------------------------|--------------------------------------------|
| 入口          | 主线程顺序取                      | N worker 从共享只读池循环取               | 主线程为本批构造 B 条 stream             | 主线程维持 B 条活跃 slot                   |
| Spotter       | 1 个                              | **每线程 1 个**                           | 1 个共享                                 | **1 个共享**                               |
| Stream 数     | 1                                 | N（每线程 1）                             | B（同一 spotter 持有）                   | B（同一 spotter 持有，稳态替换）           |
| 节奏          | 全速喂                            | 每片 `sleep(len/sr)`                      | 全速喂完所有条                           | 每 tick 各喂一片 + `sleep(chunk_seconds)`  |
| 算的时机      | 每片立刻 `decode_stream`          | 每片立刻 `decode_stream`                  | 全量喂入后 `decode_streams` 反复批算     | 每 tick 一次 `decode_streams(ready)`       |
| 终止          | pool 跑完                         | `sleep(duration)` 到点                    | `n_batches` 跑完                         | `sleep(duration)` 到点                     |
| 关键 metric   | `latency.p95` / `rtf.mean`        | `latency.p95` / `rtf_per_stream.p95`      | `throughput_calls_per_sec` / `batch_rtf` | `latency.p95` / `throughput_calls_per_sec` |

---

## 4. 输出 JSON 结构

文件名：`perf-<scene>-<backend>[-tag].json`。顶层：

```json
{
  "mode": "concurrent",
  "scene": "concurrent_c8",
  "backend": "onnx",
  "tag": "",
  "chunk_seconds": 0.5,
  "manifest": "../sherpa_eval/data/wakeword_pos.jsonl",
  "n_manifest_samples": 50,
  "env": {
    "platform": "Linux-...",
    "python": "3.11.x",
    "cpu_count": 16,
    "cpu_model": "Intel(R) Xeon(R) ...",
    "provider": "cpu",
    "num_threads": 1,
    "sherpa_onnx_version": "1.x.y"
  },
  "model": {
    "encoder": "model/encoder-....onnx",
    "keywords_threshold": 0.35,
    "keywords_score": null
  },
  "result": { ... }   // mode-specific，见下
}
```

### `result` 字段

#### `single`

| 字段                              | 含义                                       |
|-----------------------------------|--------------------------------------------|
| `n_samples`                       | 实际处理条数                               |
| `elapsed_seconds`                 | wall 总耗时                                |
| `audio_seconds_total`             | 音频时长合计                               |
| `throughput_audio_per_wall`       | xRT                                        |
| `latency_seconds.{mean,p50,p90,p95,p99,max}` | 端到端延迟分布                  |
| `rtf.{mean,p50,p95}`              | RTF 分布                                   |

#### `concurrent`

| 字段                                    | 含义                                          |
|-----------------------------------------|-----------------------------------------------|
| `concurrency` / `pacing`                | 配置                                          |
| `duration_seconds`                      | 实际跑了多久                                  |
| `n_calls_total` / `n_calls_per_thread_mean` | 总调用数 / 每线程均值                     |
| `throughput_audio_per_wall`             | 等效 xRT（N 路并行，理论上限 ≈ N）            |
| `throughput_calls_per_sec`              | 每秒完成调用数                                |
| `latency_seconds.{mean,p50,p90,p95,p99,max}` | 单路端到端延迟分布（**SLO 主指标**）     |
| `rtf_per_stream.{mean,p50,p95}`         | 单流 RTF；`> 1` 表示掉队                      |

#### `batch`

| 字段                                       | 含义                                          |
|--------------------------------------------|-----------------------------------------------|
| `batch_size` / `n_batches`                 | 批配置                                        |
| `n_calls_total`                            | `batch_size × n_batches`                      |
| `elapsed_seconds` / `audio_seconds_total`  | wall 与音频合计                               |
| `throughput_audio_per_wall`                | xRT                                           |
| `throughput_calls_per_sec`                 | 调用吞吐                                      |
| `batch_wall_seconds.{mean,p50,p95,max}`    | 单 batch 耗时分布                             |
| `per_call_latency_seconds.{mean,p50,p95}`  | 摊销延迟（**非用户等待**）                    |
| `batch_rtf.mean`                           | 批音频总时长 / 批 wall                        |

#### `batch_streaming`

| 字段                                            | 含义                                                                              |
|-------------------------------------------------|-----------------------------------------------------------------------------------|
| `batch_size` / `pacing`                         | 配置（`batch_size` 由 `--concurrency` 传入；语义 = 稳态活跃 stream 数）           |
| `duration_seconds`                              | 实际跑了多久                                                                      |
| `n_calls_total` / `audio_seconds_total`         | 总完成流数 / 音频时长合计                                                         |
| `throughput_audio_per_wall`                     | xRT（理论上限 `realtime` ≈ B，`full` 可远高）                                     |
| `throughput_calls_per_sec`                      | 每秒完成调用数                                                                    |
| `latency_seconds.{mean,p50,p90,p95,p99,max}`    | **单条端到端延迟分布（SLO 主指标）** —— 真实墙钟，非摊销值                        |
| `rtf_per_stream.{mean,p50,p95}`                 | 单流 RTF；`realtime` 下 `> 1.3` 是危险信号（缓冲在涨）                            |

---

## 5. 使用方式

### 5.1 一键跑全部场景

在 `run.sh` 顶部 `SCENES` 数组配置好后：

```bash
bash run.sh
```

### 5.2 只跑某个场景

```bash
bash run.sh --only concurrent_c8
```

### 5.3 改输出目录或单参数

```bash
bash run.sh --output-dir ../runs/exp003_xxx/metrics
bash run.sh --pacing realtime --keywords-threshold 0.30
```

### 5.4 单场景直接调 Python

```bash
# single
python sherpa_onnx_kws_perf.py \
    --tokens model/tokens.txt --encoder ... --decoder ... --joiner ... \
    --keywords-file model/keywords.txt \
    --manifest ../sherpa_eval/data/wakeword_pos.jsonl \
    --mode single --limit 50 --num-threads 1 \
    --scene single_cpu1t --output-dir ../runs/exp001_baseline/metrics

# concurrent
python sherpa_onnx_kws_perf.py ... \
    --mode concurrent --concurrency 8 --duration-seconds 30 --pacing realtime \
    --scene concurrent_c8 --output-dir ...

# batch
python sherpa_onnx_kws_perf.py ... \
    --mode batch --batch-size 16 --n-batches 20 \
    --scene batch_b16 --output-dir ...

# batch_streaming（复用 concurrent 的参数：--concurrency 即 batch_size）
python sherpa_onnx_kws_perf.py ... \
    --mode batch_streaming --concurrency 8 --duration-seconds 30 --pacing realtime \
    --scene batch_streaming_b8 --output-dir ...
```

### 5.5 `cpu_sweep`：CPU 预算下扫并发(Linux)

在指定并发点列表上,对内层 `concurrent` 或 `batch_streaming` 各跑一次,后台用 `psutil` 采样进程 CPU%,
输出 `perf-*.json` + 同名 `perf-*.csv`(供 `scripts/report.ipynb` 第 10/11 节画图)。

```bash
# 绑核 0-3，per_core 口径下 target=70%(即 70%/核;4 核上限 400%)
python sherpa_onnx_kws_perf.py \
    --tokens ... --encoder ... --decoder ... --joiner ... \
    --keywords-file ... --manifest ... \
    --mode cpu_sweep \
    --inner-mode concurrent \
    --concurrency-list "1,2,4,8,16,30,64" \
    --target-cpu 70 \
    --cpu-budget-mode per_core \
    --cpu-affinity "0-3" \
    --duration-seconds 30 \
    --pacing realtime \
    --num-threads 1 \
    --scene cpu_sweep_c_core4 --tag t70_a0-3_d30 \
    --output-dir ../runs/exp003_xxx/metrics
```

依赖:`pip install psutil`。绑核走 `os.sched_setaffinity`(仅 Linux 有效;其它平台 warn 后跳过)。

`run.sh` 里加场景:

```bash
SCENES+=("cpu_sweep_c_core4|cpu_sweep|concurrent|1,2,4,8,16,30,64|t70_a0-3_d30")
```

tag 段语义(用 `_` 分段,可选):
- `t<N>` → `--target-cpu N`(默认 70)
- `a<S>` → `--cpu-affinity S`(如 `a0` / `a0-3` / `a0,2,4`)
- `d<N>` → `--duration-seconds N`(每个并发点的时长,默认 30)
- `b<M>` → `--cpu-budget-mode M`(`per_core`|`total`,默认 `per_core`)

`cpu_budget_mode` 口径:
- `per_core`:psutil 默认行为,单核 100% = 1 核;`target=70` 即 0.7 核;
  绑 4 核时 CPU% 上限 = 400。
- `total`:除以 `cpu_count`,`target=70` = 全机算力的 70%。

---

## 6. 决策指南

| 想知道                          | 看哪个                                                              |
|---------------------------------|---------------------------------------------------------------------|
| 这块板子能跑实时吗              | `single_cpu1t` 的 `rtf.mean < 0.3` 较稳，`< 1` 才有可能              |
| ORT 多线程加速比                | `single_cpu1t` vs `single_cpu4t` 的 `throughput_audio_per_wall`     |
| 上线能开几路（独立实例部署）    | `concurrent_cN` 扫一组，找 `latency.p95` 起拐点的 N（留 20% buffer）|
| 服务端 batch 吞吐上限           | `batch_b8` vs `batch_b16` 的 `throughput_calls_per_sec` 增益是否值得 `per_call_latency` 的代价 |
| 服务端 B 路实时流的真实 SLO     | `batch_streaming_bN` 的 `latency_seconds.p95`（`pacing=realtime`），`rtf_per_stream.p95 > 1.3` 即危险 |
| 给定 X% CPU 预算最高几路并发    | `cpu_sweep` 的 `max_concurrency_under_budget`，配 `scripts/report.ipynb` 第 10/11 节出曲线           |

---

## 7. 设计取舍

- **`concurrent` 每线程独立 spotter，`batch` 与 `batch_streaming` 共享 spotter**
  - 真实多路部署每路是独立解码状态机；共享 spotter 但多 stream 与多实例的内存/缓存行为不同。
  - `batch` / `batch_streaming` 的加速正是要走 `decode_streams([...])`，必须共享同一个 spotter；差别在喂入节奏。
- **`batch_streaming` 稳态替换 vs `batch` 凑批**
  - `batch_streaming` 维持 B 条活跃 stream，任一条完成立即从音频池补一条（与 `concurrent` 稳态语义对齐），保证每个 tick 都有 ~B 条在并行 forward。
  - `batch` 模式则是 B 条流一起开始、一起结束的 cohort 语义，凑批排队时间不计入 wall——所以它的延迟数据不能直接当 SLO 看。
- **`_preload_pool` 把音频全量读入内存**
  - 把 I/O 从热路径剥离；样本数过大时受 RAM 限制，用 `--limit` 控。
- **`_tail_padding` 缓存按采样率复用**（line 149-159）
  - 避免热路径 `np.zeros` 分配影响计时稳定性。
- **`_summarize_single` 用 numpy percentile，concurrent 单流 RTF 不报 p99/max**
  - 并发尾延迟的极端点不代表系统问题，看 p95 更稳健。
- **`warmup` 默认 2 条**
  - 跳过 ORT 首次推理的 cold-start，避免 P50 被首条拖偏；超大模型可调高。
- **`pacing` 不影响 RTF 公式本身**
  - RTF 永远是 `wall / audio`；但 `realtime` 下 wall 包含 sleep，所以 RTF ≈ 1 + 算力开销 / 音频时长。
- **`cpu_sweep` 是外层编排，不重写并发逻辑**
  - 内部直接复用 `run_concurrent` / `run_batch_streaming`；只在外层加 `_CpuSampler`(后台线程 200ms 采样 `psutil.Process.cpu_percent`) 和并发点遍历。换内层算法不影响 CPU 采样。
  - 绑核用 `os.sched_setaffinity`(Linux 原生)，记入 JSON `env.affinity_cores`；非 Linux 平台 warn 后跳过。
  - `max_concurrency_under_budget` 用 `cpu_p95 ≤ target_cpu` 判定——p95 比 mean 更能反映尾部抖动是否压住核。
