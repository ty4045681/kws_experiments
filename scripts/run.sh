#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# scripts/run.sh —— 实验登记 / 入库 / 出报告的一键脚本
#
# 仿 icefall run.sh 的 stage 风格,按需跑某几步:
#   stage 1 : 新建实验目录(分配 expNNN_<name>)
#   stage 2 : 解析该实验 metrics/ 下的 metric-*.txt 与 perf-*.json
#   stage 3 : 把指标入库到 registry.csv / per_command.csv / per_perf.csv
#   stage 4 : 重建总览 REPORT.md（含准确率与性能两部分）
#   stage 5 : (可选)全库扫描 + 重建所有 CSV(忽略 EXP_DIR)
#
# 性能数据来自 sherpa_perf/run.sh 产出的 metrics/perf-*.json
#
# 用法示例
#   # 第一次:建实验目录,然后你去跑训练 + decode,把 metric-*.txt 拷进 metrics/
#   bash scripts/run.sh --stage 1 --stop-stage 1 --name lr5e-5 --variable lr --value 5e-5
#
#   # 训练好后,一键解析→入库→出报告
#   bash scripts/run.sh --stage 2 --stop-stage 4 --exp-dir runs/exp002_lr5e-5
#
#   # 改了若干历史实验后,全量重建
#   bash scripts/run.sh --stage 5 --stop-stage 5
#
# 也可改顶部默认值后直接 `bash scripts/run.sh`。
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ───── 默认参数(命令行可覆盖)────────────────────────────────────────────
stage=1
stop_stage=4

# stage 1 需要
NAME=""              # 必填(stage 1):实验短名,如 lr5e-5
VARIABLE=""          # 本次消融改的变量名,如 lr
VALUE=""             # 变量取值,如 5e-5
NOTES=""             # 备注

# stage 2/3 需要(stage 1 完成后自动捕获并写入此变量,后续 stage 沿用)
EXP_DIR=""

# Python 解释器
PYTHON=${PYTHON:-python3}

# ───── 解析命令行 ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage)        stage="$2"; shift 2 ;;
    --stop-stage)   stop_stage="$2"; shift 2 ;;
    --name)         NAME="$2"; shift 2 ;;
    --variable)     VARIABLE="$2"; shift 2 ;;
    --value)        VALUE="$2"; shift 2 ;;
    --notes)        NOTES="$2"; shift 2 ;;
    --exp-dir)      EXP_DIR="$2"; shift 2 ;;
    --python)       PYTHON="$2"; shift 2 ;;
    -h|--help)
      sed -n '2,30p' "$0"; exit 0 ;;
    *)
      echo "[error] 未知参数: $1"; exit 1 ;;
  esac
done

# ───── 切到项目根目录(本脚本所在目录的上一级) ────────────────────────────
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "$SCRIPT_DIR/.." &>/dev/null && pwd)"
cd "$ROOT"
echo "[info] ROOT = $ROOT"
echo "[info] stage = $stage, stop_stage = $stop_stage"

log() { echo -e "\n========== $* =========="; }

# ───── stage 1 : 新建实验目录 ──────────────────────────────────────────────
if [[ $stage -le 1 && $stop_stage -ge 1 ]]; then
  log "Stage 1: 新建实验目录"
  if [[ -z "$NAME" ]]; then
    echo "[error] stage 1 需要 --name"; exit 1
  fi
  # 捕获 new_experiment.py 的输出,提取生成的目录路径
  out=$("$PYTHON" scripts/new_experiment.py \
        --name "$NAME" \
        ${VARIABLE:+--variable "$VARIABLE"} \
        ${VALUE:+--value "$VALUE"} \
        ${NOTES:+--notes "$NOTES"})
  echo "$out"
  EXP_DIR="$(echo "$out" | sed -n 's|^已创建 \(.*\)$|\1|p' | head -1)"
  if [[ -z "$EXP_DIR" ]]; then
    echo "[error] 未能从输出中解析出 EXP_DIR"; exit 1
  fi
  EXP_DIR="$(realpath --relative-to="$ROOT" "$EXP_DIR" 2>/dev/null || echo "$EXP_DIR")"
  echo "[info] EXP_DIR = $EXP_DIR"
  echo
  echo "下一步:跑你的训练 + decode,把产出的 metric-*.txt 重命名后放入:"
  echo "    $EXP_DIR/metrics/"
  echo "命名约定: metric-<testset>-<backend>[-tag].txt  (backend ∈ {pt, onnx})"
  echo "完成后再跑: bash scripts/run.sh --stage 2 --exp-dir $EXP_DIR"
fi

# 后续 stage 都需要 EXP_DIR(stage 5 除外)
need_exp_dir() {
  if [[ -z "$EXP_DIR" ]]; then
    echo "[error] stage $1 需要 --exp-dir(或先跑 stage 1)"; exit 1
  fi
  if [[ ! -d "$EXP_DIR" ]]; then
    echo "[error] EXP_DIR 不存在: $EXP_DIR"; exit 1
  fi
}

# ───── stage 2 : 解析 metric-*.txt ────────────────────────────────────────
if [[ $stage -le 2 && $stop_stage -ge 2 ]]; then
  log "Stage 2: 解析 metric-*.txt 与 perf-*.json -> metrics.json"
  need_exp_dir 2
  shopt -s nullglob
  metric_files=("$EXP_DIR"/metrics/metric-*.txt)
  perf_files=("$EXP_DIR"/metrics/perf-*.json)
  shopt -u nullglob
  if [[ ${#metric_files[@]} -eq 0 && ${#perf_files[@]} -eq 0 ]]; then
    echo "[error] $EXP_DIR/metrics/ 下既没有 metric-*.txt 也没有 perf-*.json"; exit 1
  fi
  "$PYTHON" scripts/parse_decode.py "$EXP_DIR"
fi

# ───── stage 3 : 入库 registry.csv / per_command.csv ──────────────────────
if [[ $stage -le 3 && $stop_stage -ge 3 ]]; then
  log "Stage 3: 入库 registry.csv / per_command.csv / per_perf.csv"
  need_exp_dir 3
  "$PYTHON" scripts/update_registry.py "$EXP_DIR"
fi

# ───── stage 4 : 生成 REPORT.md ────────────────────────────────────────────
if [[ $stage -le 4 && $stop_stage -ge 4 ]]; then
  log "Stage 4: 生成 REPORT.md"
  "$PYTHON" scripts/build_report.py
  echo "[info] 报告位置: $ROOT/REPORT.md"
fi

# ───── stage 5 : 全库重建(不依赖 EXP_DIR)───────────────────────────────
if [[ $stage -le 5 && $stop_stage -ge 5 ]]; then
  log "Stage 5: 全库重建 CSV + 报告"
  "$PYTHON" scripts/update_registry.py --rebuild
  "$PYTHON" scripts/build_report.py
fi

echo
echo "[done] 全部完成。"
