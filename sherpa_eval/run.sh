#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# sherpa_eval/run.sh —— 用 sherpa-onnx 对自己的音频跑 KWS 评测的一键脚本
#
# 支持一次性跑多个自定义 testset(如 wakeword_pos / car_noise / kitchen 等)。
#
# stage 1 : 为每个 testset 生成 manifest
# stage 2 : 为每个 testset 跑评测,产出 metric-{TESTSET}-{SUFFIX}.txt
# stage 3 : 阈值扫描,扫一组 --keywords-threshold,产出多份 metric
#
# ──────── 配置方式 ───────────────────────────────────────────────────────
# 在脚本顶部"测试集表"里,按下面格式逐个 testset 配置:
#   TESTSETS+=("name|mode|src|extra")
# 其中:
#   - name  : testset 名(任意,不必是 small/large)
#   - mode  : transcript / auto-pair / fixed-text
#   - src   : 模式对应的源
#             * transcript  -> "/path/to/audio_dir|/path/to/transcript_file"
#             * auto-pair   -> "/path/to/audio_dir"
#             * fixed-text  -> "/path/to/audio_dir|the fixed text"
#   - extra : 额外参数(给 build_manifest.py),可空,例如 "--no-recursive"
#
# 评测脚本会给每个 testset 调一次 build_manifest 和 sherpa_onnx_kws_eval,
# 产出到统一的 --output-dir(默认就是上层某个实验的 metrics/ 目录)。
#
# ──────── 典型用法 ───────────────────────────────────────────────────────
# 1) 默认:跑所有配置的 testset(stage 1 -> 2)
#    bash sherpa_eval/run.sh
#
# 2) 只为 wakeword_pos 重做 manifest
#    bash sherpa_eval/run.sh --only wakeword_pos --stage 1 --stop-stage 1
#
# 3) 阈值扫描 stage 3(对所有 testset 都扫)
#    bash sherpa_eval/run.sh --stage 3 --stop-stage 3 \
#         --thresholds "0.20 0.25 0.30 0.35 0.40"
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ─────────────── 用户配置区 ─────────────────────────────────────────────────
stage=1
stop_stage=2

# --- 模型 ---
TOKENS="model/tokens.txt"
ENCODER="model/encoder-epoch-12-avg-2-chunk-16-left-128.onnx"
DECODER="model/decoder-epoch-12-avg-2-chunk-16-left-128.onnx"
JOINER="model/joiner-epoch-12-avg-2-chunk-16-left-128.onnx"
KEYWORDS_FILE="model/keywords.txt"

# --- manifest 输出根目录(每个 testset 一份 .jsonl)---
MANIFEST_DIR="data"

# --- 测试集表 ----------------------------------------------------------------
# 格式: "name|mode|src|extra"
# 留空数组,在下面按需追加。下面是几个示例(默认全部启用,可注释掉不要的)。
TESTSETS=()

# 示例 1:正样本批量,所有 wav 都说 "lights on"
TESTSETS+=("wakeword_pos|fixed-text|/data/wakeword_pos|lights on|")

# 示例 2:负样本,有整份 transcript
TESTSETS+=("car_noise|transcript|/data/car_noise|/data/car_noise.text|")

# 示例 3:每个 wav 旁边都有同名 .txt
TESTSETS+=("kitchen|auto-pair|/data/kitchen||")

# --- 评测参数 ---
SUFFIX="onnx"                                  # backend 后缀
OUTPUT_DIR="../runs/exp001_baseline/metrics"   # 直接落到上层脚手架的实验目录
KW_THRESHOLD="0.35"
KW_SCORE=""
CHUNK_SECONDS="0.5"
NUM_THREADS="2"
PROVIDER="cpu"
LIMIT=""
EXTRA_ARGS=""

# --- stage 3 阈值扫描 ---
THRESHOLDS="0.20 0.25 0.30 0.35 0.40"

# 只跑某一个 testset(命令行 --only 设置)
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
    --manifest-dir)     MANIFEST_DIR="$2"; shift 2 ;;
    --suffix)           SUFFIX="$2"; shift 2 ;;
    --output-dir)       OUTPUT_DIR="$2"; shift 2 ;;
    --keywords-threshold) KW_THRESHOLD="$2"; shift 2 ;;
    --keywords-score)   KW_SCORE="$2"; shift 2 ;;
    --chunk-seconds)    CHUNK_SECONDS="$2"; shift 2 ;;
    --num-threads)      NUM_THREADS="$2"; shift 2 ;;
    --provider)         PROVIDER="$2"; shift 2 ;;
    --limit)            LIMIT="$2"; shift 2 ;;
    --thresholds)       THRESHOLDS="$2"; shift 2 ;;
    --extra-args)       EXTRA_ARGS="$2"; shift 2 ;;
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
echo "[info] cwd  = $HERE"
echo "[info] stage = $stage, stop_stage = $stop_stage"

mkdir -p "$MANIFEST_DIR" "$OUTPUT_DIR"

log() { echo -e "\n========== $* =========="; }

# ─── 解析一行 TESTSETS 配置 -> 全局变量 _NAME / _MODE / _SRC / _EXTRA ────
parse_entry() {
  local entry="$1"
  IFS='|' read -r _NAME _MODE _SRC1 _SRC2 _EXTRA <<< "$entry"
  _NAME="${_NAME:-}"
  _MODE="${_MODE:-}"
  _SRC1="${_SRC1:-}"
  _SRC2="${_SRC2:-}"
  _EXTRA="${_EXTRA:-}"
  if [[ -z "$_NAME" || -z "$_MODE" ]]; then
    echo "[error] 测试集条目格式错误: '$entry'"; return 1
  fi
}

# ─── 给单条测试集跑 build_manifest.py ──────────────────────────────────────
build_one_manifest() {
  local name="$1" mode="$2" src1="$3" src2="$4" extra="$5"
  local out="$MANIFEST_DIR/${name}.jsonl"
  echo "  [build_manifest] $name (mode=$mode) -> $out"

  case "$mode" in
    transcript)
      [[ -d "$src1" && -f "$src2" ]] || { echo "[error] $name: 需要 src1=音频目录 src2=transcript 文件"; return 1; }
      # shellcheck disable=SC2086
      "$PYTHON" build_manifest.py \
          --audio-dir "$src1" \
          --transcript "$src2" \
          --output "$out" $extra
      ;;
    auto-pair)
      [[ -d "$src1" ]] || { echo "[error] $name: 需要 src1=音频目录"; return 1; }
      # shellcheck disable=SC2086
      "$PYTHON" build_manifest.py \
          --audio-dir "$src1" --auto-pair \
          --output "$out" $extra
      ;;
    fixed-text)
      [[ -d "$src1" && -n "$src2" ]] || { echo "[error] $name: 需要 src1=音频目录 src2=固定文本"; return 1; }
      # shellcheck disable=SC2086
      "$PYTHON" build_manifest.py \
          --audio-dir "$src1" \
          --fixed-text "$src2" \
          --output "$out" $extra
      ;;
    *)
      echo "[error] $name: 未知 mode='$mode' (transcript/auto-pair/fixed-text)"
      return 1
      ;;
  esac
}

# ─── 给单条测试集跑一次评测 ────────────────────────────────────────────────
run_one_eval() {
  local name="$1" threshold="$2" sub_suffix="$3"
  local manifest="$MANIFEST_DIR/${name}.jsonl"
  [[ -f "$manifest" ]] || { echo "[error] manifest 不存在: $manifest (先跑 stage 1)"; return 1; }
  local extra=""
  [[ -n "$KW_SCORE"   ]] && extra="$extra --keywords-score $KW_SCORE"
  [[ -n "$LIMIT"      ]] && extra="$extra --limit $LIMIT"
  [[ -n "$EXTRA_ARGS" ]] && extra="$extra $EXTRA_ARGS"

  echo "  [eval] testset=$name threshold=$threshold -> metric-${name}-${sub_suffix}.txt"
  # shellcheck disable=SC2086
  "$PYTHON" sherpa_onnx_kws_eval.py \
      --tokens   "$TOKENS" \
      --encoder  "$ENCODER" \
      --decoder  "$DECODER" \
      --joiner   "$JOINER" \
      --keywords-file "$KEYWORDS_FILE" \
      --manifest "$manifest" \
      --testset  "$name" \
      --suffix   "$sub_suffix" \
      --output-dir "$OUTPUT_DIR" \
      --keywords-threshold "$threshold" \
      --chunk-seconds "$CHUNK_SECONDS" \
      --num-threads "$NUM_THREADS" \
      --provider "$PROVIDER" \
      $extra
}

# ─── 校验测试集表 ──────────────────────────────────────────────────────────
if [[ ${#TESTSETS[@]} -eq 0 ]]; then
  echo "[error] TESTSETS 数组为空,请在脚本顶部至少配置一个测试集"; exit 1
fi

# 过滤 --only
declare -a ACTIVE
if [[ -n "$ONLY" ]]; then
  for e in "${TESTSETS[@]}"; do
    parse_entry "$e"
    [[ "$_NAME" == "$ONLY" ]] && ACTIVE+=("$e")
  done
  if [[ ${#ACTIVE[@]} -eq 0 ]]; then
    echo "[error] --only $ONLY 没有匹配任何测试集"; exit 1
  fi
else
  ACTIVE=("${TESTSETS[@]}")
fi
echo "[info] 即将处理 ${#ACTIVE[@]} 个测试集:"
for e in "${ACTIVE[@]}"; do parse_entry "$e"; echo "  - $_NAME ($_MODE)"; done

# ─── stage 1 : 生成 manifest(每个 testset 一份)──────────────────────────
if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
  log "Stage 1: 生成 manifest -> $MANIFEST_DIR/<name>.jsonl"
  for e in "${ACTIVE[@]}"; do
    parse_entry "$e"
    build_one_manifest "$_NAME" "$_MODE" "$_SRC1" "$_SRC2" "$_EXTRA"
  done
fi

# ─── stage 2 : 单阈值评测(每个 testset 跑一次)───────────────────────────
if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
  log "Stage 2: 评测  threshold=$KW_THRESHOLD  suffix=$SUFFIX"
  for f in "$TOKENS" "$ENCODER" "$DECODER" "$JOINER" "$KEYWORDS_FILE"; do
    [[ -f "$f" ]] || { echo "[error] 模型文件不存在: $f"; exit 1; }
  done
  for e in "${ACTIVE[@]}"; do
    parse_entry "$e"
    run_one_eval "$_NAME" "$KW_THRESHOLD" "$SUFFIX"
  done
fi

# ─── stage 3 : 阈值扫描(每个 testset × 每个阈值)─────────────────────────
if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
  log "Stage 3: 阈值扫描  thresholds = $THRESHOLDS"
  for e in "${ACTIVE[@]}"; do
    parse_entry "$e"
    echo -e "\n  --- testset = $_NAME ---"
    for t in $THRESHOLDS; do
      run_one_eval "$_NAME" "$t" "${SUFFIX}-t${t}"
    done
  done
  echo
  echo "[info] $OUTPUT_DIR/ 下现在含有多份 metric-<testset>-${SUFFIX}-tX.XX.txt"
fi

echo
echo "[done] sherpa_eval 完成 (处理 ${#ACTIVE[@]} 个测试集)。"
