#!/bin/bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-1}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-1}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-1}
export KMP_DUPLICATE_LIB_OK=${KMP_DUPLICATE_LIB_OK:-TRUE}
export PYTORCH_ENABLE_MPS_FALLBACK=${PYTORCH_ENABLE_MPS_FALLBACK:-1}

OUT_DIR=./results/ppo_quadrotor_3D_robust_figure8_completion
TASK_CONFIG=./config_overrides/quadrotor_3D/quadrotor_3D_robust_figure8_completion.yaml
MAX_ENV_STEPS=5000000
SAVE_INTERVAL=10000
LOG_INTERVAL=10000
EVAL_INTERVAL=50000
ROLLOUT_BATCH_SIZE=8
NUM_WORKERS=8
MINI_BATCH_SIZE=512
SEED=${SEED:-2}
USE_ACCELERATOR=${USE_ACCELERATOR:-1}
INIT_CHECKPOINT=""
ACTOR_LR=""
CRITIC_LR=""
DISTURBANCE_RANGE_LOW=""
DISTURBANCE_RANGE_HIGH=""
DISTURBANCE_APPLY_PROB=""
DANGER_TIME_PENALTY=""
PATH_DANGER_PENALTY=""
EXTRA_KV_OVERRIDES=()

usage() {
    cat <<EOF
Usage:
  bash ./train_robust_path_follow_ppo.sh [options]

Options:
  --out-dir PATH              Output directory for checkpoints and logs.
  --task-config PATH          Quadrotor task override YAML.
  --max-env-steps N           Total environment steps.
  --save-interval N           Save model_latest.pt every N environment steps.
  --log-interval N            Print terminal summary and append CSV every N steps.
  --eval-interval N           Run evaluation every N steps.
  --rollout-batch-size N      Number of parallel rollout environments.
  --num-workers N             Number of subprocess workers.
  --mini-batch-size N         PPO mini-batch size.
  --seed N                    Random seed.
  --init-checkpoint PATH      Start this run from an existing checkpoint if out-dir has no model_latest.pt.
  --actor-lr LR               PPO actor learning rate override.
  --critic-lr LR              PPO critic learning rate override.
  --disturbance-range LOW HIGH
                              Override impulse magnitude_range.
  --disturbance-apply-prob P  Override impulse apply_probability.
  --danger-time-penalty X     Per-step penalty for staying outside the path safety tube.
  --path-danger-penalty X     Quadratic path deviation penalty override.
  --nominal                   Disable training disturbances for nominal figure-8 completion pretraining.
  --cpu                       Force CPU instead of CUDA/MPS.
  --use-accelerator           Use CUDA/MPS when available.
  -h, --help                  Show this help.

Legacy positional form is still supported:
  OUT_DIR MAX_ENV_STEPS SAVE_INTERVAL LOG_INTERVAL EVAL_INTERVAL ROLLOUT_BATCH_SIZE NUM_WORKERS MINI_BATCH_SIZE
EOF
}

POSITIONAL=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --out-dir)
            OUT_DIR="$2"
            shift 2
            ;;
        --task-config)
            TASK_CONFIG="$2"
            shift 2
            ;;
        --max-env-steps)
            MAX_ENV_STEPS="$2"
            shift 2
            ;;
        --save-interval)
            SAVE_INTERVAL="$2"
            shift 2
            ;;
        --log-interval)
            LOG_INTERVAL="$2"
            shift 2
            ;;
        --eval-interval)
            EVAL_INTERVAL="$2"
            shift 2
            ;;
        --rollout-batch-size)
            ROLLOUT_BATCH_SIZE="$2"
            shift 2
            ;;
        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;
        --mini-batch-size)
            MINI_BATCH_SIZE="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --init-checkpoint)
            INIT_CHECKPOINT="$2"
            shift 2
            ;;
        --actor-lr)
            ACTOR_LR="$2"
            shift 2
            ;;
        --critic-lr)
            CRITIC_LR="$2"
            shift 2
            ;;
        --disturbance-range)
            DISTURBANCE_RANGE_LOW="$2"
            DISTURBANCE_RANGE_HIGH="$3"
            shift 3
            ;;
        --disturbance-apply-prob)
            DISTURBANCE_APPLY_PROB="$2"
            shift 2
            ;;
        --danger-time-penalty)
            DANGER_TIME_PENALTY="$2"
            shift 2
            ;;
        --path-danger-penalty)
            PATH_DANGER_PENALTY="$2"
            shift 2
            ;;
        --nominal)
            EXTRA_KV_OVERRIDES+=(task_config.disturbances=None)
            shift
            ;;
        --cpu)
            USE_ACCELERATOR=0
            shift
            ;;
        --use-accelerator)
            USE_ACCELERATOR=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done

if [ "${#POSITIONAL[@]}" -gt 8 ]; then
    echo "Too many positional arguments." >&2
    usage >&2
    exit 1
fi
if [ "${#POSITIONAL[@]}" -ge 1 ]; then OUT_DIR="${POSITIONAL[0]}"; fi
if [ "${#POSITIONAL[@]}" -ge 2 ]; then MAX_ENV_STEPS="${POSITIONAL[1]}"; fi
if [ "${#POSITIONAL[@]}" -ge 3 ]; then SAVE_INTERVAL="${POSITIONAL[2]}"; fi
if [ "${#POSITIONAL[@]}" -ge 4 ]; then LOG_INTERVAL="${POSITIONAL[3]}"; fi
if [ "${#POSITIONAL[@]}" -ge 5 ]; then EVAL_INTERVAL="${POSITIONAL[4]}"; fi
if [ "${#POSITIONAL[@]}" -ge 6 ]; then ROLLOUT_BATCH_SIZE="${POSITIONAL[5]}"; fi
if [ "${#POSITIONAL[@]}" -ge 7 ]; then NUM_WORKERS="${POSITIONAL[6]}"; fi
if [ "${#POSITIONAL[@]}" -ge 8 ]; then MINI_BATCH_SIZE="${POSITIONAL[7]}"; fi

if [ -n "${ACTOR_LR}" ]; then
    EXTRA_KV_OVERRIDES+=(algo_config.actor_lr="${ACTOR_LR}")
fi
if [ -n "${CRITIC_LR}" ]; then
    EXTRA_KV_OVERRIDES+=(algo_config.critic_lr="${CRITIC_LR}")
fi
if [ -n "${DANGER_TIME_PENALTY}" ]; then
    EXTRA_KV_OVERRIDES+=(task_config.path_danger_time_penalty="${DANGER_TIME_PENALTY}")
fi
if [ -n "${PATH_DANGER_PENALTY}" ]; then
    EXTRA_KV_OVERRIDES+=(task_config.path_danger_penalty="${PATH_DANGER_PENALTY}")
fi
if [ -n "${DISTURBANCE_RANGE_LOW}" ] || [ -n "${DISTURBANCE_RANGE_HIGH}" ]; then
    if [ -z "${DISTURBANCE_RANGE_LOW}" ] || [ -z "${DISTURBANCE_RANGE_HIGH}" ]; then
        echo "--disturbance-range requires LOW and HIGH." >&2
        exit 1
    fi
    DISTURBANCE_APPLY_PROB="${DISTURBANCE_APPLY_PROB:-0.5}"
    EXTRA_KV_OVERRIDES+=("task_config.disturbances={'dynamics':[{'disturbance_func':'impulse','apply_probability':${DISTURBANCE_APPLY_PROB},'magnitude_range':[${DISTURBANCE_RANGE_LOW},${DISTURBANCE_RANGE_HIGH}],'random_direction':True,'step_offset_range':[75,450],'duration':5,'decay_rate':0.85}]}")
fi

COMMON_ARGS=(
    --overrides
        ./config_overrides/quadrotor_3D/ppo_quadrotor_3D.yaml
        "${TASK_CONFIG}"
    --output_dir "${OUT_DIR}"
    --seed "${SEED}"
    --kv_overrides
        algo_config.max_env_steps="${MAX_ENV_STEPS}"
        algo_config.save_interval="${SAVE_INTERVAL}"
        algo_config.log_interval="${LOG_INTERVAL}"
        algo_config.eval_interval="${EVAL_INTERVAL}"
        algo_config.eval_save_best=True
        algo_config.rollout_batch_size="${ROLLOUT_BATCH_SIZE}"
        algo_config.num_workers="${NUM_WORKERS}"
        algo_config.mini_batch_size="${MINI_BATCH_SIZE}"
)

if [ "${#EXTRA_KV_OVERRIDES[@]}" -gt 0 ]; then
    COMMON_ARGS+=("${EXTRA_KV_OVERRIDES[@]}")
fi

if [ "${USE_ACCELERATOR}" != "0" ]; then
    COMMON_ARGS=(--use_gpu "${COMMON_ARGS[@]}")
fi

echo "Resolved robust path-following PPO training parameters:"
echo "  out_dir:              ${OUT_DIR}"
echo "  task_config:          ${TASK_CONFIG}"
echo "  max_env_steps:        ${MAX_ENV_STEPS}"
echo "  save_interval:        ${SAVE_INTERVAL}"
echo "  log_interval:         ${LOG_INTERVAL}"
echo "  eval_interval:        ${EVAL_INTERVAL}"
echo "  rollout_batch_size:   ${ROLLOUT_BATCH_SIZE}"
echo "  num_workers:          ${NUM_WORKERS}"
echo "  mini_batch_size:      ${MINI_BATCH_SIZE}"
echo "  seed:                 ${SEED}"
if [ -n "${INIT_CHECKPOINT}" ]; then
    echo "  init_checkpoint:      ${INIT_CHECKPOINT}"
else
    echo "  init_checkpoint:      (none)"
fi
if [ "${#EXTRA_KV_OVERRIDES[@]}" -gt 0 ]; then
    echo "  extra_kv_overrides:   ${EXTRA_KV_OVERRIDES[*]}"
else
    echo "  extra_kv_overrides:   (none)"
fi
echo "  use_accelerator:      ${USE_ACCELERATOR}"
echo "  OMP_NUM_THREADS:      ${OMP_NUM_THREADS}"
echo "  MKL_NUM_THREADS:      ${MKL_NUM_THREADS}"
echo "  OPENBLAS_NUM_THREADS: ${OPENBLAS_NUM_THREADS}"
echo "  PYTORCH_MPS_FALLBACK: ${PYTORCH_ENABLE_MPS_FALLBACK}"
echo

if [ -n "${INIT_CHECKPOINT}" ] && [ ! -f "${OUT_DIR}/model_latest.pt" ]; then
    if [ ! -f "${INIT_CHECKPOINT}" ]; then
        echo "Init checkpoint not found: ${INIT_CHECKPOINT}" >&2
        exit 1
    fi
    mkdir -p "${OUT_DIR}"
    SOURCE_RUN_DIR="$(dirname "${INIT_CHECKPOINT}")"
    if [ "$(basename "${SOURCE_RUN_DIR}")" = "checkpoints" ]; then
        SOURCE_RUN_DIR="$(dirname "${SOURCE_RUN_DIR}")"
    fi
    if [ ! -f "${SOURCE_RUN_DIR}/config.yaml" ]; then
        echo "Could not find source config.yaml next to checkpoint run: ${SOURCE_RUN_DIR}/config.yaml" >&2
        exit 1
    fi
    cp "${INIT_CHECKPOINT}" "${OUT_DIR}/model_latest.pt"
    cp "${SOURCE_RUN_DIR}/config.yaml" "${OUT_DIR}/config.yaml"
    echo "Initialized ${OUT_DIR} from checkpoint ${INIT_CHECKPOINT}"
    echo
fi

if [ -f "${OUT_DIR}/model_latest.pt" ]; then
    echo "Resuming robust path-following PPO from ${OUT_DIR}/model_latest.pt"
    echo "Resolved python command:"
    printf '  %q' python3 ../../safe_control_gym/experiments/train_rl_controller.py --restore "${OUT_DIR}" "${COMMON_ARGS[@]}"
    echo
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --restore "${OUT_DIR}" \
        "${COMMON_ARGS[@]}"
else
    echo "Starting fresh robust path-following PPO training in ${OUT_DIR}"
    echo "Resolved python command:"
    printf '  %q' python3 ../../safe_control_gym/experiments/train_rl_controller.py --algo ppo --task quadrotor "${COMMON_ARGS[@]}"
    echo
    python3 ../../safe_control_gym/experiments/train_rl_controller.py \
        --algo ppo \
        --task quadrotor \
        "${COMMON_ARGS[@]}"
fi

echo "Training curve CSV: ${OUT_DIR}/logs/training_curve.csv"
echo "Scalar history CSV: ${OUT_DIR}/logs/scalar_history.csv"
echo "Parallel envs: ${ROLLOUT_BATCH_SIZE}, worker processes: ${NUM_WORKERS}, mini-batch: ${MINI_BATCH_SIZE}"
echo "Accelerator enabled: ${USE_ACCELERATOR}"
