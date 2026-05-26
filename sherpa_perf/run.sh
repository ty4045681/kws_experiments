#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# sherpa_perf/run.sh —— sherpa-onnx KWS 推理性能测试一键脚本
#
# 用一个 SCENES 数组配置任意多个性能测试场景:
#   - single          : 单线程顺序解码,测时延 / RTF
#   - concurrent      : N 路并发 stream(每线程独立 spotter),测吞吐 / P95 延迟
#   - batch           : decode_streams offline 批量解码,测 batch 吞吐上限
#   - batch_streaming : B 条 stream 时间片交错喂入,共享 spotter 走 decode_streams,
#                       测真实流式 batch 的 P95 延迟与吞吐(服务端语音网关场景)
#   - cpu_sweep       : 在指定并发点列表上扫描内层 concurrent / batch_streaming,
#                       后台采样 CPU%,产 perf-*.json + perf-*.csv(供 notebook 画图)
#
# 每个场景产出一份 perf-<scene>-<backend>[-tag].json,直接落入上层脚手架
# 的实验 metrics/ 目录,被 scripts/parse_perf.py 收纳。
#
# ──────── 配置方式 ───────────────────────────────────────────────────────
# 在脚本顶部 SCENES 数组按下面 5 段加场景:
#   SCENES+=("scene|mode|arg1|arg2|tag")
# 各 mode 的 arg 含义:
#   single          arg1=<n_samples 限制,空=全部>  arg2=<num_threads(spotter)>
#   concurrent      arg1=<concurrency>             arg2=<duration_seconds>
#                                                  并配合全局 PACING=full|realtime
#   batch           arg1=<batch_size>              arg2=<n_batches>
#   batch_streaming arg1=<concurrency=batch_size>  arg2=<duration_seconds>
#                                                  并配合全局 PACING=full|realtime
#   cpu_sweep       arg1=<inner_mode: concurrent|batch_streaming>
#                   arg2=<concurrency_list: 逗号分隔, 如 "1,2,4,8,16,30,64">
#                   tag 中可编码 cpu 预算与绑核(用 '_' 分段):
#                     t<N>   : --target-cpu N    (默认 70)
#                     a<S>   : --cpu-affinity S  (例 a0 / a0-3 / a0,2,4)
#                     d<N>   : --duration-seconds N (每个并发点的持续时长, 默认 30)
#                     b<M>   : --cpu-budget-mode M (per_core|total, 默认 per_core)
#                   例: tag="t70_a0-3_d30" 表示 target=70% / 绑 0-3 / 每点 30s
# tag 是可选后缀,合并进文件名,如 -realtime / -bs16
#
# ──────── 用法 ───────────────────────────────────────────────────────────
# 跑全部场景:
#   bash sherpa_perf/run.sh
# 只跑某个场景:
#   bash sherpa_perf/run.sh --only concurrent_c8
# 改输出目录:
#   bash sherpa_perf/run.sh --output-dir ../runs/exp003_xxx/metrics
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─────────────── 用户配置区 ─────────────────────────────────────────────────
stage=1
stop_stage=1

# --- 模型 ---
TOKENS="model/tokens.txt"
ENCODER="model/encoder-epoch-12-avg-2-chunk-16-left-128.onnx"
DECODER="model/decoder-epoch-12-avg-2-chunk-16-left-128.onnx"
JOINER="model/joiner-epoch-12-avg-2-chunk-16-left-128.onnx"
KEYWORDS_FILE="model/keywords.txt"

# --- 数据(复用 sherpa_eval 的 manifest)---
MANIFEST="../sherpa_eval/data/wakeword_pos.jsonl"

# --- 通用推理参数 ---
SUFFIX="onnx"
CHUNK_SECONDS="0.5"
PROVIDER="cpu"
NUM_THREADS_DEFAULT="2"          # spotter 内 ORT 线程数
KW_THRESHOLD="0.35"
PACING="full"                    # concurrent 默认全速;改 realtime 模拟实时流
WARMUP="2"

# --- 输出目录(默认落到 baseline 实验)---
OUTPUT_DIR="../runs/exp001_baseline/metrics"

# --- 场景表 ----------------------------------------------------------------
# 格式: "scene|mode|arg1|arg2|tag"
SCENES=()

# 字段顺序: scene_name | mode | arg1 | arg2 | tag
# 各 mode 的 arg1 / arg2 含义见上方"配置方式"注释。

# 1) 单线程基线               arg1=limit(样本数=50)  arg2=num_threads(=1)
SCENES+=("single_cpu1t|single|50|1|")

# 2) 单实例 4 线程,看 ORT intra-op 加速
#                             arg1=limit(=50)        arg2=num_threads(=4)
SCENES+=("single_cpu4t|single|50|4|")

# 3) 4 路并发(每路独立 spotter,各 1 线程)
#                             arg1=concurrency(=4)   arg2=duration_seconds(=30)
SCENES+=("concurrent_c4|concurrent|4|30|")

# 4) 8 路并发                 arg1=concurrency(=8)   arg2=duration_seconds(=30)
SCENES+=("concurrent_c8|concurrent|8|30|")

# 5) batch_size=8 的批量解码  arg1=batch_size(=8)    arg2=n_batches(=20)
SCENES+=("batch_b8|batch|8|20|")

# 6) batch_size=16            arg1=batch_size(=16)   arg2=n_batches(=20)
SCENES+=("batch_b16|batch|16|20|")

# 7) 真实流式 batch (8 路并发,共享 spotter,decode_streams 批解码)
#                             arg1=concurrency=batch_size(=8)
#                             arg2=duration_seconds(=30)
SCENES+=("batch_streaming_b8|batch_streaming|8|30|")

# 8) CPU 预算扫描 - concurrent 内层, 绑 core 0-3, target 70%/核, 每点 30s
#                             arg1=inner_mode(=concurrent)
#                             arg2=concurrency_list(=1,2,4,8,16,30,64)
#                             tag=t<target>_a<aff>_d<dur>_b<budget_mode>
# SCENES+=("cpu_sweep_c_core4|cpu_sweep|concurrent|1,2,4,8,16,30,64|t70_a0-3_d30")
# SCENES+=("cpu_sweep_bs_core4|cpu_sweep|batch_streaming|1,2,4,8,16,30,64|t70_a0-3_d30")
# SCENES+=("cpu_sweep_c_core1|cpu_sweep|concurrent|1,2,4,8,16,30|t70_a0_d30")

# --- 只跑某个场景(命令行 --only)---
ONLY=""

PYTHON=${PYTHON:-python3}
# ─────────────── 用户配置区结束 ────────────────────────────────────────────

# ─── 解析命令行覆盖 ───────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)            stage="$2"; shift 2 ;;
    --stop-stage)       stop_stage="$2"; shift 2 ;;
    --tokens)           TOKENS="$2"; shift 2 ;;
    --encoder)          ENCODER="$2"; shift 2 ;;
    --decoder)          DECODER="$2"; shift 2 ;;
    --joiner)           JOINER="$2"; shift 2 ;;
    --keywords-file)    KEYWORDS_FILE="$2"; shift 2 ;;
    --manifest)         MANIFEST="$2"; shift 2 ;;
    --suffix)           SUFFIX="$2"; shift 2 ;;
    --chunk-seconds)    CHUNK_SECONDS="$2"; shift 2 ;;
    --provider)         PROVIDER="$2"; shift 2 ;;
    --keywords-threshold) KW_THRESHOLD="$2"; shift 2 ;;
    --pacing)           PACING="$2"; shift 2 ;;
    --warmup)           WARMUP="$2"; shift 2 ;;
    --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
    --only)             ONLY="$2"; shift 2 ;;
    --python)           PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,40p' "$0"; exit 0 ;;
    *)
      echo "[error] 未知参数: $1"; exit 1 ;;
  esac
done

# ─── 进入脚本所在目录 ─────────────────────────────────────────────────────
HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
cd "$HERE"
echo "[info] cwd        = $HERE"
echo "[info] manifest   = $MANIFEST"
echo "[info] output_dir = $OUTPUT_DIR"

mkdir -p "$OUTPUT_DIR"

log() { echo -e "\n========== $* =========="; }

# ─── 解析一行 SCENES -> _SC_NAME _SC_MODE _SC_A1 _SC_A2 _SC_TAG ───────────
parse_scene() {
  local entry="$1"
  IFS='|' read -r _SC_NAME _SC_MODE _SC_A1 _SC_A2 _SC_TAG <<< "$entry"
  _SC_NAME="${_SC_NAME:-}"
  _SC_MODE="${_SC_MODE:-}"
  _SC_A1="${_SC_A1:-}"
  _SC_A2="${_SC_A2:-}"
  _SC_TAG="${_SC_TAG:-}"
  if [[ -z "$_SC_NAME" || -z "$_SC_MODE" ]]; then
    echo "[error] 场景格式错误: '$entry'"; return 1
  fi
}

# ─── 跑单个场景 ────────────────────────────────────────────────────────────
run_one_scene() {
  local name="$1" mode="$2" a1="$3" a2="$4" tag="$5"
  local common=(
    --tokens   "$TOKENS"
    --encoder  "$ENCODER"
    --decoder  "$DECODER"
    --joiner   "$JOINER"
    --keywords-file "$KEYWORDS_FILE"
    --manifest "$MANIFEST"
    --chunk-seconds "$CHUNK_SECONDS"
    --provider "$PROVIDER"
    --keywords-threshold "$KW_THRESHOLD"
    --output-dir "$OUTPUT_DIR"
    --scene    "$name"
    --suffix   "$SUFFIX"
    --warmup   "$WARMUP"
  )
  [[ -n "$tag" ]] && common+=(--tag "$tag")

  case "$mode" in
    single)
      local limit_arg=()
      [[ -n "$a1" ]] && limit_arg=(--limit "$a1")
      local nt="${a2:-$NUM_THREADS_DEFAULT}"
      echo "  [perf] $name single  limit=${a1:-all}  num_threads=$nt"
      "$PYTHON" sherpa_onnx_kws_perf.py \
          "${common[@]}" --mode single --num-threads "$nt" "${limit_arg[@]}"
      ;;
    concurrent)
      local conc="${a1:?需要 concurrency}"
      local dur="${a2:-30}"
      echo "  [perf] $name concurrent  N=$conc  duration=${dur}s  pacing=$PACING"
      "$PYTHON" sherpa_onnx_kws_perf.py \
          "${common[@]}" --mode concurrent \
          --num-threads 1 \
          --concurrency "$conc" --duration-seconds "$dur" --pacing "$PACING"
      ;;
    batch)
      local bs="${a1:?需要 batch_size}"
      local nb="${a2:-20}"
      echo "  [perf] $name batch  batch_size=$bs  n_batches=$nb"
      "$PYTHON" sherpa_onnx_kws_perf.py \
          "${common[@]}" --mode batch \
          --num-threads "$NUM_THREADS_DEFAULT" \
          --batch-size "$bs" --n-batches "$nb"
      ;;
    batch_streaming)
      local conc="${a1:?需要 concurrency (= batch_size)}"
      local dur="${a2:-30}"
      echo "  [perf] $name batch_streaming  N=$conc  duration=${dur}s  pacing=$PACING"
      "$PYTHON" sherpa_onnx_kws_perf.py \
          "${common[@]}" --mode batch_streaming \
          --num-threads "$NUM_THREADS_DEFAULT" \
          --concurrency "$conc" --duration-seconds "$dur" --pacing "$PACING"
      ;;
    cpu_sweep)
      local inner="${a1:?需要 inner_mode (concurrent|batch_streaming)}"
      local conc_list="${a2:?需要 concurrency_list (e.g. '1,2,4,8,16')}"
      # 解析 tag: t<target>_a<affinity>_d<duration>_b<budget_mode>; 各段可选
      local target_cpu="70"
      local affinity=""
      local dur="30"
      local budget="per_core"
      if [[ -n "$tag" ]]; then
        IFS='_' read -ra _parts <<< "$tag"
        for _p in "${_parts[@]}"; do
          case "$_p" in
            t*) target_cpu="${_p#t}" ;;
            a*) affinity="${_p#a}" ;;
            d*) dur="${_p#d}" ;;
            b*) budget="${_p#b}" ;;
          esac
        done
      fi
      echo "  [perf] $name cpu_sweep  inner=$inner  conc_list=$conc_list  "\
"target=${target_cpu}% (${budget})  aff=${affinity:-<none>}  dur=${dur}s  pacing=$PACING"
      local aff_arg=()
      [[ -n "$affinity" ]] && aff_arg=(--cpu-affinity "$affinity")
      "$PYTHON" sherpa_onnx_kws_perf.py \
          "${common[@]}" --mode cpu_sweep \
          --num-threads 1 \
          --inner-mode "$inner" \
          --concurrency-list "$conc_list" \
          --target-cpu "$target_cpu" \
          --cpu-budget-mode "$budget" \
          --duration-seconds "$dur" \
          --pacing "$PACING" \
          "${aff_arg[@]}"
      ;;
    *)
      echo "[error] $name: 未知 mode='$mode' (single/concurrent/batch/batch_streaming/cpu_sweep)"
      return 1
      ;;
  esac
}

# ─── 校验场景表 ────────────────────────────────────────────────────────────
if [[ ${#SCENES[@]} -eq 0 ]]; then
  echo "[error] SCENES 数组为空,请在脚本顶部至少配置一个场景"; exit 1
fi

# 过滤 --only
declare -a ACTIVE
if [[ -n "$ONLY" ]]; then
  for e in "${SCENES[@]}"; do
    parse_scene "$e"
    [[ "$_SC_NAME" == "$ONLY" ]] && ACTIVE+=("$e")
  done
  if [[ ${#ACTIVE[@]} -eq 0 ]]; then
    echo "[error] --only $ONLY 没有匹配任何场景"; exit 1
  fi
else
  ACTIVE=("${SCENES[@]}")
fi
echo "[info] 即将处理 ${#ACTIVE[@]} 个场景:"
for e in "${ACTIVE[@]}"; do parse_scene "$e"; echo "  - $_SC_NAME ($_SC_MODE)"; done

# ─── stage 1 : 跑所有场景 ──────────────────────────────────────────────────
if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
  log "Stage 1: 运行性能测试  $(date '+%Y-%m-%d %H:%M:%S')"
  for f in "$TOKENS" "$ENCODER" "$DECODER" "$JOINER" "$KEYWORDS_FILE" "$MANIFEST"; do
    [[ -f "$f" ]] || { echo "[error] 文件不存在: $f"; exit 1; }
  done
  for e in "${ACTIVE[@]}"; do
    parse_scene "$e"
    run_one_scene "$_SC_NAME" "$_SC_MODE" "$_SC_A1" "$_SC_A2" "$_SC_TAG"
  done
fi

echo
echo "[done] sherpa_perf 完成 (处理 ${#ACTIVE[@]} 个场景)。"
echo "      产出文件位于 $OUTPUT_DIR/perf-*.json"
echo "      下一步:在脚手架根目录跑 scripts/run.sh --stage 2 入库"
