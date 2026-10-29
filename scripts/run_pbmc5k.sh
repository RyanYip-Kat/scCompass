#!/usr/bin/env bash
set -euo pipefail

# -------------------------- 脚本信息 & 工具函数 --------------------------
SCRIPT_NAME="$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 颜色定义（非终端时自动失效）
if [[ -t 1 ]]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    NC='\033[0m' # No Color
else
    RED=''
    GREEN=''
    YELLOW=''
    NC=''
fi

info() {
    echo -e "${GREEN}[INFO] $(date '+%Y-%m-%d %H:%M:%S') $*${NC}"
}
warn() {
    echo -e "${YELLOW}[WARN] $(date '+%Y-%m-%d %H:%M:%S') $*${NC}"
}
error() {
    echo -e "${RED}[ERROR] $(date '+%Y-%m-%d %H:%M:%S') $*${NC}" >&2
}

usage() {
cat <<EOF
${SCRIPT_NAME}: 一键运行 scCompass 主pipeline + 可视化报告

用法:
  ${SCRIPT_NAME} [OPTIONS]

环境变量方式传参（推荐，和原来保持兼容）：
  示例：INPUT=xxx.h5ad SOLVER_BACKEND=gurobi FORCE=1 ./${SCRIPT_NAME}

Options:
  -h, --help          打印本帮助并退出

可配置参数（环境变量，括号内为默认值）：
  PYTHON              python解释器路径 (./.venv/bin/python)
  INPUT               输入h5ad文件 (testdata/pbmc5k_querydata.h5ad)
  PIPELINE_OUTPUT     pipeline结果目录 (outputs/pbmc5k_full_pipeline)
  VIS_OUTPUT          可视化图表输出目录 (outputs/pbmc5k_visual_report)
  MODEL_SCOPE         模型范围 (central-carbon)
  SOLVER_BACKEND      求解器后端 (cplex)
  NUM_PROCESSES       并行进程数 (8)
  NUM_THREADS         单进程线程数 (1)
  MICROCLUSTER_SIZE   微聚类大小 (1000)
  MICROCLUSTER_FILTER 微聚类过滤策略 (novar)
  CHECKPOINT_BATCH_SIZE 断点续跑batch大小 (16)
  GROUPBY             绘图分组列名 (celltype)
  TOP_N               绘图展示topN条目 (30)
  FIGURE_FORMAT       图片格式 png/pdf/svg (png)
  FORCE=1             开启--overwrite，覆盖已有结果 (0)
  RESET_CHECKPOINT=1  开启--overwrite-checkpoint，清除断点 (0)
  EXPORT_SCORES=1     可视化时导出score表格 (0)

流程：
  1. 运行 scripts/pipeline.py 主计算流程
  2. 运行 scripts/pipeline_vis.py 生成图表&汇总表
EOF
}

# 捕获help参数
if [[ $# -ge 1 && ($1 == "-h" || $1 == "--help") ]]; then
    usage
    exit 0
fi

# -------------------------- 参数配置区（可修改） --------------------------
# 所有默认值集中在这里，方便查阅修改
PYTHON="${PYTHON:-${PROJECT_ROOT}/.venv/bin/python}"
INPUT="${INPUT:-testdata/pbmc5k_querydata.h5ad}"
PIPELINE_OUTPUT="${PIPELINE_OUTPUT:-outputs/pbmc5k_full_pipeline}"
VIS_OUTPUT="${VIS_OUTPUT:-outputs/pbmc5k_visual_report}"
MODEL_SCOPE="${MODEL_SCOPE:-central-carbon}"
SOLVER_BACKEND="${SOLVER_BACKEND:-cplex}"
NUM_PROCESSES="${NUM_PROCESSES:-8}"
NUM_THREADS="${NUM_THREADS:-1}"
MICROCLUSTER_SIZE="${MICROCLUSTER_SIZE:-1000}"
MICROCLUSTER_FILTER="${MICROCLUSTER_FILTER:-novar}"
CHECKPOINT_BATCH_SIZE="${CHECKPOINT_BATCH_SIZE:-16}"
GROUPBY="${GROUPBY:-celltype}"
VOLCANO_GROUP_A="${VOLCANO_GROUP_A:-CD4 Naive}"
VOLCANO_GROUP_B="${VOLCANO_GROUP_B:-CD4 Tem}"
VOLCANO_Y_CAP="${VOLCANO_Y_CAP:-20}"
TOP_N="${TOP_N:-30}"
FIGURE_FORMAT="${FIGURE_FORMAT:-png}"
FORCE="${FORCE:-0}"
RESET_CHECKPOINT="${RESET_CHECKPOINT:-0}"
EXPORT_SCORES="${EXPORT_SCORES:-0}"

# -------------------------- 前置检查 & 打印配置 --------------------------
info "项目根目录: ${PROJECT_ROOT}"
cd "${PROJECT_ROOT}"

info "===== 当前生效全部配置 ====="
cat <<CONF
PYTHON              = ${PYTHON}
INPUT               = ${INPUT}
PIPELINE_OUTPUT     = ${PIPELINE_OUTPUT}
VIS_OUTPUT          = ${VIS_OUTPUT}
MODEL_SCOPE         = ${MODEL_SCOPE}
SOLVER_BACKEND      = ${SOLVER_BACKEND}
NUM_PROCESSES       = ${NUM_PROCESSES}
NUM_THREADS         = ${NUM_THREADS}
MICROCLUSTER_SIZE   = ${MICROCLUSTER_SIZE}
MICROCLUSTER_FILTER = ${MICROCLUSTER_FILTER}
CHECKPOINT_BATCH_SIZE = ${CHECKPOINT_BATCH_SIZE}
GROUPBY             = ${GROUPBY}
TOP_N               = ${TOP_N}
FIGURE_FORMAT       = ${FIGURE_FORMAT}
FORCE               = ${FORCE}
RESET_CHECKPOINT    = ${RESET_CHECKPOINT}
EXPORT_SCORES       = ${EXPORT_SCORES}
CONF
info "============================"

# 简单校验
if [[ ! -x "${PYTHON}" ]]; then
    error "Python不存在或不可执行: ${PYTHON}"
    exit 1
fi
if [[ ! -f "${INPUT}" ]]; then
    error "输入h5ad文件不存在: ${INPUT}"
    exit 1
fi

# -------------------------- 第一步：运行主pipeline --------------------------
info "[1/2] 启动 scCompass pipeline 计算"
pipeline_args=(
  --input "${INPUT}"
  --output-dir "${PIPELINE_OUTPUT}"
  --model-scope "${MODEL_SCOPE}"
  --layer counts
  --solver-backend "${SOLVER_BACKEND}"
  --num-processes "${NUM_PROCESSES}"
  --num-threads "${NUM_THREADS}"
  --microcluster-size "${MICROCLUSTER_SIZE}"
  --microcluster-filter "${MICROCLUSTER_FILTER}"
  --checkpoint-batch-size "${CHECKPOINT_BATCH_SIZE}"
  --plot-top-reactions 4
)

[[ "${FORCE}" == "1" ]] && pipeline_args+=(--overwrite)
[[ "${RESET_CHECKPOINT}" == "1" ]] && pipeline_args+=(--overwrite-checkpoint)

"${PYTHON}" scripts/pipeline.py "${pipeline_args[@]}"
info "[1/2] pipeline 计算完成"

# -------------------------- 第二步：可视化报告 --------------------------
info "[2/2] 开始生成可视化图表、汇总表格"
vis_args=(
  --input "${PIPELINE_OUTPUT}/result.h5ad"
  --output-dir "${VIS_OUTPUT}"
  --groupby "${GROUPBY}"
  --volcano-group-a "${VOLCANO_GROUP_A}"
  --volcano-group-b "${VOLCANO_GROUP_B}"
  --volcano-y-cap "${VOLCANO_Y_CAP}"
  --top-n "${TOP_N}"
  --format "${FIGURE_FORMAT}"
)
[[ "${EXPORT_SCORES}" == "1" ]] && vis_args+=(--export-scores)

"${PYTHON}" scripts/pipeline_vis.py "${vis_args[@]}"
info "[2/2] 可视化报告完成，结果目录：${VIS_OUTPUT}"

info "===== 全部任务执行成功 ====="

